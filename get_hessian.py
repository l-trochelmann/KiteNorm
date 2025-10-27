"""Pretrain a Transformer on language modeling."""

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

  # Load checkpoint and starting step
  ckpt, micro_step_start = maybe_load_checkpoint(cfg, device)

  # Dataset
  trainloader, validloader = get_dataloaders(cfg)
  
  # Model
  model, model_cfg = construct_model(cfg)
  model = model.to(device)

  # Hessian at initialisation
  print_master(f"=== Approximating Hessian ESD... ===")
  model.eval()

  # Disable flash attention
  for layer in model.layers:
      if hasattr(layer, "attn"):
          layer.attn.track_entropy = True

  # Select training data subset
  subset_batches = cfg.esd_batches  # 128 samples with 1024 seq_len
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

  # Compute
  wandb.init(project="LN-variants", name=cfg.wandb_run_name, config=cfg._asdict())

  hess = Hessian(model, criterion_hess, dataloader=subset, cuda=str(device).startswith('cuda'))
  iter_steps, n_v = 100, 10
  density_eigs, density_wts = hess.density(iter=iter_steps, n_v=n_v)

  max_eigs, _ = hess.eigenvalues(maxIter=200, tol=1e-4, top_n=1)
  lambda_max = float(max_eigs[0])

  # Save
  artifact = wandb.Artifact(f"hessian-esd_{cfg.wandb_run_name}", type="hessian-esd")

  with tempfile.TemporaryDirectory() as td:
    npz_path = os.path.join(td, "hessian_esd.npz")
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

  # DDP slaughtering
  destroy_ddp()


if __name__ == "__main__":
  app.run(main)
