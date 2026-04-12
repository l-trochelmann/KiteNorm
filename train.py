"""Pretrain a Transformer on language modeling."""

from absl import app, flags
from collections import defaultdict

import utils
import torch
import math
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


def _resolve_layer_scaled(v, n_layers: int):
  if isinstance(v, str):
    s = v.strip().lower().replace(" ", "")
    if s == "2l":
      return float(2 * n_layers)
    if s == "1/sqrt(l)":
      return 1 / math.sqrt(n_layers)
    if s == "1/l":
      return float(1.0 / n_layers)
    if s == "1/(2l)":
      return 1.0 / float(2 * n_layers)
  return v


def main(_):
  print("=== Prepare Training... ===")
  
  CFG_PATH, JOB_IDX = FLAGS.config, FLAGS.job_idx
  cfg, _ = utils.load_config(CFG_PATH, JOB_IDX)

  # (overcome .yaml limitation: automatic skip scale and res-scale)
  cfg = cfg._replace(
    skip_scale=_resolve_layer_scaled(cfg.skip_scale, cfg.n_layers),
    res_scale=_resolve_layer_scaled(cfg.res_scale, cfg.n_layers),
  )
  
  local_rank, world_size, device, master_process = pytorch_setup(cfg)

  print("Using device:", device)
  print("torch.cuda.is_available():", torch.cuda.is_available())
  if torch.cuda.is_available():
      print("CUDA device count:", torch.cuda.device_count())
      print("Current CUDA device:", torch.cuda.current_device())
      print("GPU name:", torch.cuda.get_device_name(torch.cuda.current_device()))
  
  if master_process:
    utils.maybe_make_dir(cfg, JOB_IDX)

  if cfg.use_wandb and master_process:
    utils.init_wandb(cfg)
  
  # Load checkpoint and starting step
  ckpt, micro_step_start = maybe_load_checkpoint(cfg, device)

  # Dataset
  trainloader, validloader = get_dataloaders(cfg, micro_step_start)
  
  # Model
  model, model_cfg = construct_model(cfg)
  
  # Engine
  engine = TorchEngine(model, cfg, device, local_rank, ckpt)
  if cfg.regulariser:
    is_var_regulariser = cfg.regulariser in {"var_L1", "var_ReLU"}
    is_alignment_regulariser = cfg.regulariser == "alignment_ReLU"
    for m in model.modules():
      if type(m).__name__ == "VarCollector":
        m.regularise_var = is_var_regulariser
        m.regularise_alignment = is_alignment_regulariser
        m.regularise = is_var_regulariser or is_alignment_regulariser

  # Initialise trackers:
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
  sublayer_tracking = getattr(cfg, "track_sublayer_variance", False) or getattr(cfg, "track_sublayer_kurtosis", False) or getattr(cfg, "track_token_alignment", False)

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
    train_loss = engine.step(micro_batch)
    train_losses.append(train_loss)

    # Eval
    valid_loss = None
    if cfg.eval and ((just_updated and step % cfg.eval_every_steps == 0) or micro_step==1):
      print_master("Evaluating on validation set")

      if cfg.track_softmax:  # Enable running average before eval
        for layer in model.layers:
          layer.attn.track_entropy = True

      if sublayer_tracking:  # Enable running averages before eval
        collector_attrs = (
            "coll_attn_in", "coll_attn_out", "coll_attn_add",
            "coll_mlp_in",  "coll_mlp_out",  "coll_mlp_add",
        )
        for layer in model.layers:
            for attr in collector_attrs:
                if getattr(cfg, "track_sublayer_variance", False):
                  getattr(layer, attr).track_variance = True
                if getattr(cfg, "track_sublayer_kurtosis", False):
                  getattr(layer, attr).track_kurtosis = True
                if getattr(cfg, "track_token_alignment", False):
                  getattr(layer, attr).track_alignment = True

      valid_loss = engine.eval(validloader)  # Run eval

      if cfg.track_softmax:  # Disable running average after eval
        for layer in model.layers:
          layer.attn.track_entropy = False

      if sublayer_tracking:  # Disable running averages after eval
        for layer in model.layers:
            for attr in collector_attrs:
                if getattr(cfg, "track_sublayer_variance", False):
                  getattr(layer, attr).track_variance = False
                if getattr(cfg, "track_sublayer_kurtosis", False):
                  getattr(layer, attr).track_kurtosis = False
                if getattr(cfg, "track_token_alignment", False):
                  getattr(layer, attr).track_alignment = False
  
    # Log
    if (just_updated and step % cfg.log_every_steps == 0) or micro_step==1:
      utils.log(cfg, metrics, micro_step, train_losses, valid_loss, engine.optimizer, world_size, model=model, init_logits=init_logits, 
                probe_inputs=probe_inputs, ctx=engine.ctx, init_params=init_params, reg_term=engine.reg_term,
                master_process=master_process)
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
  valid_loss = None

  if cfg.eval:
    print_master("Evaluating on validation set (final)")
    if cfg.track_softmax:  # Enable running average before eval
      for layer in model.layers:
        layer.attn.track_entropy = True

    if sublayer_tracking:  # Enable running averages before eval
      collector_attrs = (
          "coll_attn_in", "coll_attn_out", "coll_attn_add",
          "coll_mlp_in",  "coll_mlp_out",  "coll_mlp_add",
      )
      for layer in model.layers:
          for attr in collector_attrs:
              if getattr(cfg, "track_sublayer_variance", False):
                getattr(layer, attr).track_variance = True
              if getattr(cfg, "track_sublayer_kurtosis", False):
                getattr(layer, attr).track_kurtosis = True
              if getattr(cfg, "track_token_alignment", False):
                getattr(layer, attr).track_alignment = True
  
    valid_loss = engine.eval(validloader)  # Run eval

    if cfg.track_softmax:  # Disable running average after eval
      for layer in model.layers:
        layer.attn.track_entropy = False

    if sublayer_tracking:  # Disable running averages after eval
      for layer in model.layers:
          for attr in collector_attrs:
              if getattr(cfg, "track_sublayer_variance", False):
                getattr(layer, attr).track_variance = False
              if getattr(cfg, "track_sublayer_kurtosis", False):
                getattr(layer, attr).track_kurtosis = False
              if getattr(cfg, "track_token_alignment", False):
                getattr(layer, attr).track_alignment = False

  utils.log(cfg, metrics, micro_step, train_losses, valid_loss, engine.optimizer, world_size, model=model, init_logits=init_logits, 
            probe_inputs=probe_inputs, ctx=engine.ctx, init_params=init_params, reg_term=engine.reg_term,
            master_process=master_process)
  if master_process and cfg.save_last_checkpoint:
    save_checkpoint(micro_step-1, model, engine, cfg, JOB_IDX)

  print_master(f"=== Terminating... ===")

  # DDP slaughtering
  destroy_ddp()


if __name__ == "__main__":
  app.run(main)
