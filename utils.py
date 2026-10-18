import os
import yaml
import math
import shutil
import wandb
import torch

from itertools import product
from collections import namedtuple


def load_config(path, job_idx=None):
  """
  Parse a yaml file and return the correspondent config as a namedtuple.
  If the config files has multiple entries, returns the one corresponding to job_idx.
  """
  
  with open(path, 'r') as file:
    config_dict = yaml.safe_load(file)
  Config = namedtuple('Config', config_dict.keys())

  if job_idx is None:
    cfg = config_dict
    sweep_size = 1

  else:
    keys = list(config_dict.keys())
    values = [val if isinstance(val, list) else [val] for val in config_dict.values()]
    combinations = list(product(*values))

    sweep_size = len(combinations)
    if job_idx >= sweep_size:
      raise ValueError("job_idx exceeds the total number of hyperparam combinations.")

    combination = combinations[job_idx]
    cfg = {keys[i]: combination[i] for i in range(len(keys))}
  
  return Config(**cfg), sweep_size


def init_wandb(cfg):
  """Initalizes a wandb run"""
  os.environ["WANDB__SERVICE_WAIT"] = "600"
  os.environ["WANDB_SILENT"] = "true"
  wandb.init(
    project=cfg.wandb_project, 
    name=cfg.wandb_run_name, 
    dir=cfg.wandb_dir,
    config=cfg._asdict()
  )


def maybe_make_dir(cfg, job_idx=None):
  """Creates an experiment directory if checkpointing is enabled"""
  if not cfg.save_intermediate_checkpoints and not cfg.save_last_checkpoint:
    return
  if cfg.resume and cfg.resume_exp_name is None:  # if resuming from the same exp
    return

  exp_dir = os.path.join(cfg.out_dir, cfg.exp_name)
  if job_idx is not None:  # subfolder for each job in the sweep
    exp_dir = os.path.join(exp_dir, f"job_idx_{job_idx}")

  if os.path.exists(exp_dir):
    if not cfg.over_write:
      raise ValueError(f"Found existing exp_dir at {exp_dir}.")
    print(f"Removing experiment dir: {exp_dir}")
    shutil.rmtree(exp_dir)

  print(f"Creating experiment directory: {exp_dir}")
  os.makedirs(exp_dir, exist_ok=True)
  with open(os.path.join(exp_dir, 'config.yaml'), 'w') as file:
    yaml.dump(cfg._asdict(), file, default_flow_style=False)


def compute_grad_norms(model):
  """Computes gradient l2 norms for all parameters.
    
  Returns:
      dict: A dictionary mapping parameter names to their gradient norms
  """
  grad_norms = {}
  for name, param in model.named_parameters():
    if param.grad is None:
      continue
    with torch.no_grad():
      grad_norms[f"grad_l2-norm/{name}"] = param.grad.norm(p=2).item()
  return grad_norms


def get_ln_param_stats(model):
  """Tracks normalisation-related affine parameters.

  Returns:
      dict: A dictionary mapping parameter names to scalar values or mean/std stats.
  """
  ln_param_stats = {}
  with torch.no_grad():
    for name, param in model.named_parameters():
      if "norm.weight" in name or "norm_in.weight" in name or "norm_out.weight" in name:
        if param.numel() == 1:
          ln_param_stats[f"LN_gain/{name}"] = param.data.item()
        else:
          ln_param_stats[f"LN_gain_mean/{name}"] = param.data.mean().item()
          ln_param_stats[f"LN_gain_std/{name}"] = param.data.std().item()
      elif "norm.bias" in name:
        if param.numel() == 1:
          ln_param_stats[f"LN_bias/{name}"] = param.data.item()
        else:
          ln_param_stats[f"LN_bias_mean/{name}"] = param.data.mean().item()
          ln_param_stats[f"LN_bias_std/{name}"] = param.data.std().item()

  return ln_param_stats


def get_sublayer_variance(model, cfg):
    """
    Retrieves mean hidden-state variance at each collection point (via running average)
    for each block/layer. Resets collector running stats after reading.
    If some sublayer branches are rescaled, additionally reports the rescaled variance.

    Returns:
        dict: A dicitionary mapping collection point names to calculated variance.
    """
    hidden_variances = {}
    model.eval()

    ddp = torch.distributed.is_available() and torch.distributed.is_initialized()

    skip_scale = getattr(cfg, "skip_scale", -1)
    res_scale = getattr(cfg, "res_scale", -1)

    collector_attrs = (
        "coll_attn_in", "coll_attn_out", "coll_attn_add",
        "coll_mlp_in",  "coll_mlp_out",  "coll_mlp_add",
    )

    for layer in model.layers:
        lid = layer.layer_id

        # Fetch and log collected vars
        raw_avg = {}
        for attr in collector_attrs:
            coll = getattr(layer, attr)

            var_sum = coll.running_var_sum.clone()
            count = coll.running_count_1.clone()

            if ddp:
                torch.distributed.all_reduce(var_sum, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(count, op=torch.distributed.ReduceOp.SUM)

            avg = (var_sum / count).item() if count.item() > 0 else float("nan")
            raw_avg[attr] = avg

            name = attr.replace("coll_", "").replace("_", "-")
            hidden_variances[f"hidden_variance_{name}/{lid}"] = avg

            coll.running_var_sum.zero_()
            coll.running_count_1.zero_()

        # Log rescaled vars if they exist
        if skip_scale != -1:
          hidden_variances[f"hidden_variance_attn-in_scaled-skip/{lid}"]  = skip_scale**2 * raw_avg["coll_attn_in"]
          hidden_variances[f"hidden_variance_mlp-in_scaled-skip/{lid}"]   = skip_scale**2 * raw_avg["coll_mlp_in"]
        if res_scale != -1:
          hidden_variances[f"hidden_variance_attn-out_scaled-res/{lid}"]  = res_scale**2  * raw_avg["coll_attn_out"]
          hidden_variances[f"hidden_variance_mlp-out_scaled-res/{lid}"]   = res_scale**2  * raw_avg["coll_mlp_out"]

    return hidden_variances


def get_token_alignment(model):
    """
    Like get_sublayer_variance, but to retrieve cosine alignemnt and non-mean portion.
    Supports DDP by reducing sums and counts across ranks.
    """
    alignment_metrics = {}
    model.eval()

    ddp = torch.distributed.is_available() and torch.distributed.is_initialized()

    collector_attrs = (
        "coll_attn_add",
        "coll_mlp_add",
    )

    for layer in model.layers:
        lid = layer.layer_id

        for attr in collector_attrs:
            coll = getattr(layer, attr, None)
            if coll is None or not getattr(coll, "is_after_add", False):
                continue

            cos_sum = coll.running_token_cos_alignment_sum.clone()
            non_mean_sum = coll.running_token_non_mean_portion_sum.clone()
            count = coll.running_count_3.clone()

            if ddp:
                torch.distributed.all_reduce(cos_sum, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(non_mean_sum, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(count, op=torch.distributed.ReduceOp.SUM)

            cos_avg = (cos_sum / count).item() if count.item() > 0 else float("nan")
            non_mean_avg = (non_mean_sum / count).item() if count.item() > 0 else float("nan")

            name = attr.replace("coll_", "").replace("_", "-")
            alignment_metrics[f"token_cos-alignment_{name}/{lid}"] = cos_avg
            alignment_metrics[f"token_non-mean-portion_{name}/{lid}"] = non_mean_avg

            coll.running_token_cos_alignment_sum.zero_()
            coll.running_token_non_mean_portion_sum.zero_()
            coll.running_count_3.zero_()

    return alignment_metrics


def log(cfg, metrics, micro_step, train_losses, valid_loss, optimizer, world_size, model=None, reg_term=None, master_process=True):
  "Computes new metrics and appends them to metrics. Logs on wandb. Prints log."
  # NOTE: valid_loss is already reduced across GPUs in engine.eval()

  ddp = torch.distributed.is_available() and torch.distributed.is_initialized()

  train_loss = None
  if isinstance(train_losses, list) and train_losses:
    train_loss = torch.stack(train_losses).mean()
    if ddp:
      torch.distributed.all_reduce(train_loss, op=torch.distributed.ReduceOp.SUM)
      train_loss = train_loss / world_size
    train_loss = train_loss.item()

  reg_term_log = None
  if reg_term is not None:
    reg_term_log = reg_term.detach().clone()
    if ddp:
      torch.distributed.all_reduce(reg_term_log, op=torch.distributed.ReduceOp.SUM)
      reg_term_log = reg_term_log / world_size

  sublayer_variances = None
  if getattr(cfg, "track_sublayer_variance", False) and valid_loss is not None:
    sublayer_variances = get_sublayer_variance(model, cfg)

  token_alignment = None
  if getattr(cfg, "track_token_alignment", False) and valid_loss is not None:
    token_alignment = get_token_alignment(model)

  if not master_process:
    return

  new_metrics = {
    "micro_step": micro_step,
    "step": int(micro_step / cfg.grad_accumulation_steps),
    "tokens": micro_step * cfg.micro_batch_size * cfg.seq_len * world_size,
    "lr": optimizer.param_groups[0].get("lr", float("NaN")),
  }
  new_metrics["train/reg_term"] = reg_term_log.item() if reg_term_log is not None else None
  new_metrics["train/loss"] = train_loss
  new_metrics["train/ppl"] = math.exp(train_loss) if train_loss is not None and train_loss < 709.78 else float("inf")

  if valid_loss is not None:
    new_metrics["valid/loss"] = valid_loss
    new_metrics["valid/ppl"] = math.exp(valid_loss) if valid_loss < 709.78 else float("inf")

  if getattr(cfg, "track_grad_norm", False):
    new_metrics.update(compute_grad_norms(model))

  if getattr(cfg, "track_ln_weights", False):
    new_metrics.update(get_ln_param_stats(model))

  if sublayer_variances is not None:
    new_metrics.update(sublayer_variances)

  if token_alignment is not None:
    new_metrics.update(token_alignment)

  for k,v in new_metrics.items():
    metrics[k].append(v)

  if cfg.print_progress:
    msg = ' | '.join(
      f"{key}: {value:.3e}" if isinstance(value, float) else f"{key}: {value}"
      for key, value in new_metrics.items()
    )
    print(msg)
  
  if cfg.use_wandb:
    wandb.log(new_metrics, step=new_metrics["micro_step"])


def print_master(msg):
  """Prints only in master process if using multiple GPUs."""
  rank = os.environ.get('RANK', -1)
  ddp = int(rank) != -1
  master_process = (not ddp) or (int(rank) == 0)
  
  if master_process:
    print(msg)
