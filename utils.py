import os
import yaml
import math
import shutil
import wandb
import torch
import torch.nn.functional as F

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
  """Computes gradient l2 norms for the last layer of each MLP.
    
  Returns:
      dict: A dictionary mapping layer names to their gradient norms
  """
  grad_norms = {}
  for name, param in model.named_parameters():
    if param.grad is None or not name.endswith("mlp.fc2.weight"):
      continue
    with torch.no_grad():
      grad_norms[f"grad_l2-norm/{name}"] = param.grad.norm(p=2).item()
  return grad_norms


def compute_model_update_l2(model, init_logits, probe_inputs, ctx):
  """Computes the model update according to the DeepNet paper.
  The token outputs are flattened and directly subtracted.

  Returns:
      dict: A ditionary with one key that maps to ||F(x, theta) - F(x, theta_0)||_2
  """
  model.eval()
  with ctx, torch.no_grad():
    new_logits = getattr(model(probe_inputs), 'logits', model(probe_inputs))
  update_norm = (new_logits - init_logits).flatten().norm(p=2).item()
  return {"model_update/l2-cumulative": update_norm}


def compute_model_update_cosine(model, init_logits, probe_inputs, ctx):
  """Computes the model update inspired by the DeepNet paper, but uses cosine distance rather than l2 distance.
  The cosine similarity is computed token-wise and we retrieve the mean.

  Returns:
      dict: A ditionary with one key that maps to 1 - cos_sim(F(x, theta), F(x, theta_0))
  """
  model.eval()
  with ctx, torch.no_grad():
    new_logits = getattr(model(probe_inputs), 'logits', model(probe_inputs))
    a = init_logits.view(-1, init_logits.size(-1))  # Flatten to (B*L, D)
    b = new_logits.view(-1, new_logits.size(-1))  # Flatten to (B*L, D)
  cos_sim = F.cosine_similarity(a, b, dim=1).mean().item()
  cos_dist = 1.0 - cos_sim
  return {"model_update/cosine-cumulative": cos_dist}


def compute_param_update_l2(model, init_params):
  """Computes gradient l2 norms for the last layer of each MLP.
    
  Returns:
      dict: A dictionary mapping layer names to their cumulative l2 parameter update
  """
  l2_updates = {}
  with torch.no_grad():
    for name, param in model.named_parameters():
      if not name.endswith("mlp.fc2.weight"):
        continue
      diff_norm = (param.data - init_params[name]).norm(p=2).item()
      l2_updates[f"param_update_l2-cumulative/{name}"] = diff_norm
  return l2_updates


def compute_param_update_cos(model, init_params):
  """Computes gradient l2 norms for the last layer of each MLP.
    
  Returns:
      dict: A dictionary mapping layer names to their cumulative cosine parameter update
  """
  cos_updates = {}
  with torch.no_grad():
    for name, param in model.named_parameters():
      if not name.endswith("mlp.fc2.weight"):
          continue
      a = init_params[name].flatten()
      b = param.data.flatten()
      # F.cosine_similarity expects tensors with an explicit dim
      cos_sim = F.cosine_similarity(a, b, dim=0).item()
      cos_updates[f"param_update_cosine-cumulative/{name}"] = 1.0 - cos_sim
  return cos_updates


def get_softmax_entropy(model):
  """Retrieves mean entropy of the softmax activations over the validation set for each layer."""
  softmax_entropies = {}
  model.eval()
  for layer in model.layers:
    layer_softmax_entropy = layer.attn.entropy_sum / layer.attn.entropy_count
    softmax_entropies[f"softmax_entropy/{layer.layer_id}"] = layer_softmax_entropy.item()

    layer.attn.entropy_sum.zero_()  # Reset running average before the next val pass
    layer.attn.entropy_count.zero_()  # Reset running average before the next val pass
  
  return softmax_entropies

def log(cfg, metrics, micro_step, train_losses, valid_loss, optimizer, world_size, model=None, init_logits=None, probe_inputs=None, ctx=None, init_params=None):
  "Computes new metrics and appends them to metrics. Logs on wandb. Prints log."
  # NOTE: train_losses is an array of losses, if DDP, this is from master_process only
  # NOTE: valid_loss is a float, already reduced across GPUs

  if isinstance(train_losses, list):
    train_loss = torch.stack(train_losses).mean().item() # avg loss

  new_metrics = {
    "micro_step": micro_step,
    "step": int(micro_step / cfg.grad_accumulation_steps),
    "tokens": micro_step * cfg.micro_batch_size * cfg.seq_len * world_size,
    "lr": optimizer.param_groups[0].get("lr", float("NaN")),
    "train/loss": train_loss,
    "train/ppl": math.exp(train_loss),
  }
  if valid_loss is not None:
    new_metrics["valid/loss"] = valid_loss
    new_metrics["valid/ppl"] = math.exp(valid_loss)
  
  # Add gradient norms if requested
  if cfg.track_grad_norm:
    grad_norms = compute_grad_norms(model)
    new_metrics.update(grad_norms)

  # Add model update metrics if requested
  if cfg.track_model_update:
    mu_l2 = compute_model_update_l2(model, init_logits, probe_inputs, ctx)
    mu_cos = compute_model_update_cosine(model, init_logits, probe_inputs, ctx)
    new_metrics.update(mu_l2)
    new_metrics.update(mu_cos)

  # Add param update metrics if requested
  if cfg.track_param_update:
    pu_l2 = compute_param_update_l2(model, init_params)
    pu_cos = compute_param_update_cos(model, init_params)
    new_metrics.update(pu_l2)
    new_metrics.update(pu_cos)

  # Add softmax entropy metrics if requested, only following a validation pass
  if cfg.track_softmax and valid_loss is not None:
    new_metrics.update(get_softmax_entropy(model))

  for k,v in new_metrics.items():
    metrics[k].append(v)

  if cfg.print_progress:
    msg = ' | '.join(
      f"{key}: {value:.3e}" if isinstance(value, float) else f"{key}: {value}"
      for key, value in new_metrics.items()
    )
    print(msg)
  
  if cfg.use_wandb:
    wandb.log(new_metrics)


def print_master(msg):
  """Prints only in master process if using multiple GPUs."""
  rank = os.environ.get('RANK', -1)
  ddp = int(rank) != -1
  master_process = (not ddp) or (int(rank) == 0)
  
  if master_process:
    print(msg)
