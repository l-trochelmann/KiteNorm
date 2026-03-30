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