"""Custom regularisers for improving/replacing LayerNorm"""

import torch
import torch.nn.functional as F


def var_L1(model):
  """
  Mean of L1 distances between current variances and target variances
  """
  which_collectors = (  # Use collectors just before normalisation
            # "coll_attn_in", 
            # "coll_attn_out", 
            "coll_attn_add",
            # "coll_mlp_in",  
            # "coll_mlp_out",  
            "coll_mlp_add",
        )
  penalties = []
  for layer in model.layers:
    for name in which_collectors:
      coll = getattr(layer, name, None)
      if coll is not None:
        penalty = torch.abs(coll.last_var - 1.0).mean()  # average tokenwise L1 distance
        penalties.append(penalty)

  if not penalties:
    raise NotImplementedError("Attempting to fetch sublayer variances for the regulariser, but no specified VarCollector was found.")
  
  reg = torch.stack(penalties).mean()  # average over collectors for depth-invariance
  
  return reg


def var_ReLU(model):
  """
  Mean of ReLU(X-1) where X is variance
  """
  which_collectors = (  # Use collectors just before normalisation
            # "coll_attn_in", 
            # "coll_attn_out", 
            "coll_attn_add",
            # "coll_mlp_in",  
            # "coll_mlp_out",  
            "coll_mlp_add",
        )
  penalties = []
  for layer in model.layers:
    for name in which_collectors:
      coll = getattr(layer, name, None)
      if coll is not None:
        penalty = F.relu(coll.last_var - 1.0).mean()
        penalties.append(penalty)

  if not penalties:
    raise NotImplementedError("Attempting to fetch sublayer variances for the regulariser, but no specified VarCollector was found.")
  
  reg = torch.stack(penalties).mean()  # average over collectors for depth-invariance
  
  return reg


def var_ReLU_MeanStd(model):
  """
  Mean of ReLU(X-1) where X is variance, plus std of the
  mean hidden variance across attention adds and MLP adds separately.
  """
  which_collectors = (  # Use collectors just before normalisation
            # "coll_attn_in", 
            # "coll_attn_out", 
            "coll_attn_add",
            # "coll_mlp_in",  
            # "coll_mlp_out",  
            "coll_mlp_add",
        )
  penalties = []
  mean_vars_attn = []
  mean_vars_mlp = []
  for layer in model.layers:
    for name in which_collectors:
      coll = getattr(layer, name, None)
      if coll is not None:
        penalties.append(F.relu(coll.last_var - 1.0).mean())
        if name == "coll_attn_add":
          mean_vars_attn.append(coll.last_var.mean())
        elif name == "coll_mlp_add":
          mean_vars_mlp.append(coll.last_var.mean())

  if not penalties:
    raise NotImplementedError("Attempting to fetch sublayer variances for the regulariser, but no specified VarCollector was found.")
  
  var_heterogenity = 0.0
  if mean_vars_attn:
    var_heterogenity = var_heterogenity + torch.stack(mean_vars_attn).std(unbiased=False)
  if mean_vars_mlp:
    var_heterogenity = var_heterogenity + torch.stack(mean_vars_mlp).std(unbiased=False)
  reg = torch.stack(penalties).mean() + var_heterogenity  # additional penalty for heterogeneity
  
  return reg


def alignment_ReLU(model):
  """
  Mean of ReLU(X-1) where X is non-mean portion over tokens
  """
  which_collectors = (  # Use collectors just before normalisation
            # "coll_attn_in", 
            # "coll_attn_out", 
            "coll_attn_add",
            # "coll_mlp_in",  
            # "coll_mlp_out",  
            "coll_mlp_add",
        )
  penalties = []
  for layer in model.layers:
    for name in which_collectors:
      coll = getattr(layer, name, None)
      if coll is not None:
        penalty = F.relu(0.8 - coll.last_non_mean_portion).mean()
        penalties.append(penalty)

  if not penalties:
    raise NotImplementedError("Attempting to fetch sublayer statistics for the regulariser, but no specified VarCollector was found.")
  
  reg = torch.stack(penalties).mean()  # average over collectors for depth-invariance
  
  return reg
