"""Pretrain a Transformer on language modeling, then compute Hessian ESD."""

from absl import app, flags
# from collections import defaultdict  # unused

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
# from dataclasses import asdict  # unused
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

  # ====== Train with spike detection from the start; terminate on first spike or when budget ends ======
  print_master("=== Training with live spike detection (no eval/logging) ===")
  updates_target = 512
  updates_done = 0
  micro_step = micro_step_start
  train_iter = iter(trainloader)

  prev_update_loss = None   # average loss over previous update window
  update_loss_accum = 0.0   # accumulates micro-step losses within the current update
  subset_batches = cfg.esd_batches

  # Helper to compute & log Hessian ESD for a given state dict (CPU) and tag
  def compute_and_log_esd(trained_model, state_cpu, tag, subset, update_step, loss_value, iter_steps=100, n_v=10):
    trained_model.load_state_dict(state_cpu, strict=True)
    trained_model.eval()
    if hasattr(trained_model, "layers"):
      for layer in trained_model.layers:
        if hasattr(layer, "attn"):
          layer.attn.track_entropy = True
    class LMCELoss32(torch.nn.Module):
      def forward(self, outputs, targets):
        logits = getattr(outputs, "logits", outputs)
        return torch.nn.functional.cross_entropy(
          logits.float().reshape(-1, logits.size(-1)),
          targets.reshape(-1)
        )
    criterion_hess = LMCELoss32()
    hess = Hessian(trained_model, criterion_hess, dataloader=subset, cuda=str(device).startswith('cuda'))
    density_eigs, density_wts = hess.density(iter=iter_steps, n_v=n_v)
    max_eigs, _ = hess.eigenvalues(maxIter=200, tol=1e-4, top_n=1)
    lambda_max = float(max_eigs[0])
    artifact = wandb.Artifact(
      f"hessian-esd_{cfg.wandb_run_name}_{tag}",
      type="hessian-esd",
      metadata={"update_step": int(update_step), "train_loss": float(loss_value)}
    )
    with tempfile.TemporaryDirectory() as td:
      npz_path = os.path.join(td, f"{cfg.wandb_run_name}_{tag}.npz")
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
        subset_batches=subset_batches,
        update_step=int(update_step),
        train_loss=float(loss_value),
      )
      artifact.add_file(npz_path)
      wandb.log_artifact(artifact)
    # normalise VRAM for next call
    del hess
    if (hasattr(device, "type") and device.type == "cuda") or str(device).startswith("cuda"):
      torch.cuda.empty_cache(); torch.cuda.synchronize()

  while updates_done < updates_target:
    micro_step += 1
    try:
      micro_batch = next(train_iter)
    except StopIteration:
      train_iter = iter(trainloader)
      micro_batch = next(train_iter)

    will_update = (micro_step % cfg.grad_accumulation_steps == 0)

    # Snapshot PRE-STEP weights (CPU) if this micro-step will perform the optimiser step
    if will_update:
      live_model = engine.model.module if hasattr(engine.model, "module") else engine.model
      pre_state_cpu = {k: v.detach().cpu().clone() for k, v in live_model.state_dict().items()}

    # One micro step
    loss_val = engine.step(micro_batch)
    loss_float = float(loss_val)
    update_loss_accum += loss_float

    if will_update:
      # Compute update-mean loss and advance
      curr_update_loss = update_loss_accum / float(cfg.grad_accumulation_steps)
      print("train loss: " + str(curr_update_loss))
      update_loss_accum = 0.0

      updates_done += 1
      engine.optimizer.zero_grad(set_to_none=True)

      # Spike check: compare against previous update's mean loss
      if prev_update_loss is not None and abs(curr_update_loss - prev_update_loss) > 10.0:
        print_master("=== Large loss jump detected; computing pre/post Hessian ESD ===")
        trained_model = engine.model.module if hasattr(engine.model, "module") else engine.model
        post_state_cpu = {k: v.detach().cpu().clone() for k, v in trained_model.state_dict().items()}

        # Free trainer state before Hessian
        if hasattr(engine, "scaler"):
          del engine.scaler
        del engine.optimizer
        for p in trained_model.parameters():
          p.grad = None
        if (hasattr(device, "type") and device.type == "cuda") or str(device).startswith("cuda"):
          torch.cuda.empty_cache(); torch.cuda.synchronize()

        # Build Hessian subset (smaller batch if provided)
        hess_cfg = cfg._replace(micro_batch_size=cfg.hess_micro_batch_size) if hasattr(cfg, "hess_micro_batch_size") else cfg
        hess_trainloader, _ = get_dataloaders(hess_cfg)
        subset = []
        it = iter(hess_trainloader)
        for _ in range(subset_batches):
          b = next(it)
          inp = b["input_ids"][:, :cfg.seq_len]
          tgt = b["input_ids"][:, 1:cfg.seq_len + 1]
          subset.append((inp, tgt))

        # W&B only for Hessian computations
        wandb.init(project="LN-variants", name=cfg.wandb_run_name, config=cfg._asdict())
        update_idx = micro_step // cfg.grad_accumulation_steps
        compute_and_log_esd(trained_model, pre_state_cpu,  "pre",  subset, update_idx, prev_update_loss)
        compute_and_log_esd(trained_model, post_state_cpu, "post", subset, update_idx, curr_update_loss)
        wandb.finish()
        break  # stop on first spike

      prev_update_loss = curr_update_loss

  print_master(f"=== Training ended (spike or budget reached): {updates_done} updates ===")


  print_master(f"=== Terminating... ===")
  destroy_ddp()


if __name__ == "__main__":
  app.run(main)
