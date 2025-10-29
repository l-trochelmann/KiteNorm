"""Pretrain a Transformer on language modeling, then compute Hessian ESD."""

from absl import app, flags
from collections import defaultdict

import utils
import torch
from utils import print_master
from torch_utils import pytorch_setup, destroy_ddp
from data import get_dataloaders
from checkpoint_utils import save_checkpoint, maybe_load_checkpoint
from models import construct_model
from engine import TorchEngine
from engine.engine import _move_to_device

from non_transformer.PyHessian.pyhessian import hessian as Hessian
from dataclasses import asdict
import tempfile
import os
import wandb
import numpy as np

flags.DEFINE_string('config', 'config/config.yaml', 'Path to config.yaml file.')
flags.DEFINE_integer('job_idx', None, 'Job idx for job-array sweeps. From 0 to n-1.')
FLAGS = flags.FLAGS


def main(_):

  CFG_PATH, JOB_IDX = FLAGS.config, FLAGS.job_idx
  cfg, _ = utils.load_config(CFG_PATH, JOB_IDX)

  local_rank, world_size, device, master_process = pytorch_setup(cfg)

  if master_process:
    utils.maybe_make_dir(cfg, JOB_IDX)

  # Load checkpoint and starting micro step (for grad accumulation accounting)
  ckpt, micro_step_start = maybe_load_checkpoint(cfg, device)

  # Dataset (initial loaders for training)
  trainloader, validloader = get_dataloaders(cfg)

  # Model + engine (reuse train.py’s engine; no eval/logging here)
  model, model_cfg = construct_model(cfg)
  engine = TorchEngine(model, cfg, device, local_rank, ckpt)

  # === Train for exactly 256 optimizer updates (not micro-steps) ===
  print_master("=== Training (no eval/logging) ===")
  updates_target = 512  # Hessian ESD is computed after this step.
  updates_done = 0
  micro_step = micro_step_start
  train_iter = iter(trainloader)

  while updates_done < updates_target:
    micro_step += 1
    try:
      micro_batch = next(train_iter)
    except StopIteration:
      # Re-create iterator if the loader is not infinite
      train_iter = iter(trainloader)
      micro_batch = next(train_iter)

    # One micro step; engine handles accumulation internally
    _ = engine.step(micro_batch)

    # Count an *optimizer update* when we hit the accumulation boundary
    if micro_step % cfg.grad_accumulation_steps == 0:
      updates_done += 1
      # Flush grads like train.py
      engine.optimizer.zero_grad(set_to_none=True)

  print_master(f"=== Finished training: {updates_done} updates ===")

  # Use the trained model for Hessian — unwrap if engine wrapped with DDP
  trained_model = engine.model.module if hasattr(engine.model, "module") else engine.model

  # Free up memory
  opt = engine.optimizer
  if hasattr(engine, "scaler"):
      del engine.scaler
  del opt
  del engine
  for p in trained_model.parameters():
      p.grad = None
  torch.cuda.empty_cache()
  torch.cuda.synchronize()

  # Override micro batch size for Hessian phase and rebuild loaders
  if hasattr(cfg, "hess_micro_batch_size"):
    cfg = cfg._replace(micro_batch_size=cfg.hess_micro_batch_size)
    trainloader, validloader = get_dataloaders(cfg)
  else:
    raise ValueError("Missing attribute: hess_micro_batch_size")

  # === Hessian ESD (original logic) ===
  print_master(f"=== Approximating Hessian ESD... ===")
  trained_model.eval()

  # Disable flash attention / enable entropy tracking
  if hasattr(trained_model, "layers"):
    for layer in trained_model.layers:
      if hasattr(layer, "attn"):
        layer.attn.track_entropy = True

  # Select training data subset (kept on CPU)
  subset_batches = cfg.esd_batches  # target: 128 samples with 1024 seq_len
  subset = []
  it = iter(trainloader)
  for _ in range(subset_batches):
    batch = next(it)
    # Slice next-token LM inputs/targets (keep on CPU)
    inp = batch["input_ids"][:, :cfg.seq_len]
    tgt = batch["input_ids"][:, 1:cfg.seq_len + 1]
    subset.append((inp, tgt))

  # Full precision loss
  class LMCELoss32(torch.nn.Module):
      def forward(self, outputs, targets):
          logits = getattr(outputs, "logits", outputs)
          return torch.nn.functional.cross_entropy(
              logits.float().reshape(-1, logits.size(-1)),
              targets.reshape(-1)
          )
  criterion_hess = LMCELoss32()

  # Compute + log only from here (no train-phase wandb)
  wandb.init(project="LN-variants", name=cfg.wandb_run_name, config=cfg._asdict())

  # Important: pass the trained (possibly DDP-unwrapped) model
  hess = Hessian(trained_model, criterion_hess, dataloader=subset, cuda=str(device).startswith('cuda'))
  iter_steps, n_v = 100, 10
  density_eigs, density_wts = hess.density(iter=iter_steps, n_v=n_v)

  max_eigs, _ = hess.eigenvalues(maxIter=200, tol=1e-4, top_n=1)
  lambda_max = float(max_eigs[0])

  # Save to W&B as before
  artifact = wandb.Artifact(f"hessian-esd_{cfg.wandb_run_name}", type="hessian-esd")

  with tempfile.TemporaryDirectory() as td:
    npz_path = os.path.join(td, f"{cfg.wandb_run_name}.npz")
    np.savez_compressed(
        npz_path,
        eigs=np.asarray(density_eigs, dtype=float),
        wts=np.abs(np.real_if_close(np.asarray(density_wts))),
        lambda_max=float(lambda_max),
        iter=iter_steps,
        n_v=n_v,
        arch="transformer",
        model_name=os.path.splitext(os.path.basename(CFG_PATH))[0],
        seed=int(getattr(cfg, "seed", 0)),
        subset_batches=subset_batches
    )
    artifact.add_file(npz_path)
    wandb.log_artifact(artifact)

  wandb.finish()

  print_master(f"=== Terminating... ===")

  # DDP cleanup
  destroy_ddp()


if __name__ == "__main__":
  app.run(main)
