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

flags.DEFINE_string('config', 'config/config.yaml', 'Path to config.yaml file.')
flags.DEFINE_integer('job_idx', None, 'Job idx for job-array sweeps. From 0 to n-1.')
FLAGS = flags.FLAGS


def main(_):
  
  CFG_PATH, JOB_IDX = FLAGS.config, FLAGS.job_idx
  cfg, _ = utils.load_config(CFG_PATH, JOB_IDX)
  
  local_rank, world_size, device, master_process = pytorch_setup(cfg)
  
  if master_process:
    utils.maybe_make_dir(cfg, JOB_IDX)

  if cfg.use_wandb and master_process:
    utils.init_wandb(cfg)
  
  # Load checkpoint and starting step
  ckpt, micro_step_start = maybe_load_checkpoint(cfg, device)

  # Dataset
  trainloader, validloader = get_dataloaders(cfg)
  
  # Model
  model, model_cfg = construct_model(cfg)
  
  # Engine
  engine = TorchEngine(model, cfg, device, local_rank, ckpt)


  # Initialise model update tracking:
  if not cfg.track_model_update:
    probe_inputs = None
    init_logits = None
  else:
    probe_batch = next(iter(trainloader))
    probe_inputs, _ = _move_to_device(probe_batch, cfg.seq_len, device)
    model.eval()
    with engine.ctx, torch.no_grad():
      init_logits = getattr(model(probe_inputs),'logits', model(probe_inputs)).cpu()
    init_logits = init_logits.detach()
  if not cfg.track_param_update:
    init_params = None
  else:
    init_params = {n: p.detach().cpu() for n, p in model.named_parameters()}

  # Ensure valid hidden state tracking:
  if cfg.track_normalisation_variance:
    if not cfg.ln_config in ('pre-norm', 'post-norm'):
      raise NotImplementedError("When cfg.track_normalisation_variance is True, ln_config must be either 'pre-norm' or 'post-norm'")
  if cfg.track_sublayer_variance:
    probe_batch = next(iter(trainloader))
    probe_inputs, _ = _move_to_device(probe_batch, cfg.seq_len, device)

  # Training
  print_master("=== Start Training! ===")
  metrics = defaultdict(list)
  train_losses = []

  for micro_step, micro_batch in enumerate(trainloader, micro_step_start+1):
    step = micro_step // cfg.grad_accumulation_steps
    just_updated = (micro_step % cfg.grad_accumulation_steps == 0)
    if step > cfg.steps_budget:
      break

    # Train
    if cfg.use_variance_regulariser:
      model_has_target_norm = False
      for m in model.modules():
        if type(m).__name__ in {"LayerNorm"}:
          setattr(m, "keep_last_var", True)
          model_has_target_norm = True
      if model_has_target_norm == False:
        raise NotImplementedError("Attempted to enable variance regulariser, but model has no normalisation!")
    train_loss = engine.step(micro_batch)
    train_losses.append(train_loss)

    # Eval
    valid_loss = None
    if cfg.eval and ((just_updated and step % cfg.eval_every_steps == 0) or micro_step==1):
      print_master("Evaluating on validation set")
      if cfg.track_softmax:  # Enable running average for eval
        for layer in model.layers:
          layer.attn.track_entropy = True
      if cfg.track_normalisation_variance:
        for layer in model.layers:
          layer.attn_norm.track_variance = True
          layer.mlp_norm.track_variance = True

      valid_loss = engine.eval(validloader)  # Run eval

      if cfg.track_softmax:  # Disable running average after eval
        for layer in model.layers:
          layer.attn.track_entropy = False
      if cfg.track_normalisation_variance:
        for layer in model.layers:
          layer.attn_norm.track_variance = False
          layer.mlp_norm.track_variance = False

    # Log
    if (just_updated and step % cfg.log_every_steps == 0) or micro_step==1:
      if master_process:
        utils.log(cfg, metrics, micro_step, train_losses, valid_loss, engine.optimizer, world_size, model=model, init_logits=init_logits, 
                  probe_inputs=probe_inputs, ctx=engine.ctx, init_params=init_params)
      if micro_step != 1:
        train_losses = []

    # Flush the gradients
    if just_updated:
      engine.optimizer.zero_grad(set_to_none=True) 

    # Checkpoint
    if master_process and cfg.save_intermediate_checkpoints \
        and micro_step % cfg.save_every_steps == 0:
      save_checkpoint(micro_step-1, model, engine, cfg, JOB_IDX)

  # End of training: final eval, log and save checkpoint
  print_master(f"=== Training Completed! ===")
  if master_process:
    if cfg.eval:
      print_master("Evaluating on validation set (final)")
      if cfg.track_softmax:  # Enable running average for eval
        for layer in model.layers:
          layer.attn.track_entropy = True
      if cfg.track_normalisation_variance:
        for layer in model.layers:
          layer.attn_norm.track_variance = True
          layer.mlp_norm.track_variance = True

      valid_loss = engine.eval(validloader)  # Run eval

      if cfg.track_softmax:  # Disable running average after eval
        for layer in model.layers:
          layer.attn.track_entropy = False
      if cfg.track_normalisation_variance:
        for layer in model.layers:
          layer.attn_norm.track_variance = False
          layer.mlp_norm.track_variance = False

    utils.log(cfg, metrics, micro_step, train_losses, valid_loss, engine.optimizer, world_size, model=model, init_logits=init_logits, 
              probe_inputs=probe_inputs, ctx=engine.ctx, init_params=init_params)
    if cfg.save_last_checkpoint:
      save_checkpoint(micro_step-1, model, engine, cfg, JOB_IDX)

  print_master(f"=== Terminating... ===")

  # DDP slaughtering
  destroy_ddp()


if __name__ == "__main__":
  app.run(main)
