"""Custom regularisers for improving/replacing LayerNorm"""

import torch

def mean_norm_var(model):
    vars_ = []
    for m in model.modules():
      if type(m).__name__ in {"LayerNorm"}:
        v = m.last_var
        vars_.append(v)
    if not vars_:
      raise NotImplementedError("Attempted to fetch normalisation variances for the regulariser, but model has no normalisation layers!")
    vars = torch.stack(vars_)  # shape: (n_norm_layers,)
    reg = vars.mean()
    return reg

# Var-1 target functions to implement:
# L1
# L2
# x - log(x) -1
# x^2 - log(x^2) -1

def mean_L1(model):
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


def mean_L2(model):
  """
  Sum of L2 distances between current variances and target variances
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
        penalty = torch.square(coll.last_var - 1.0).mean()  # average tokenwise L1 distance
        penalties.append(penalty)

  if not penalties:
    raise NotImplementedError("Attempting to fetch sublayer variances for the regulariser, but no specified VarCollector was found.")
  
  reg = torch.stack(penalties).mean()  # average over collectors for depth-invariance
  
  return reg


def mean_L2(model):
  """
  Sum of L2 distances between current variances and target variances
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
        penalty = torch.square(coll.last_var - 1.0).mean()  # average tokenwise L1 distance
        penalties.append(penalty)

  if not penalties:
    raise NotImplementedError("Attempting to fetch sublayer variances for the regulariser, but no specified VarCollector was found.")
  
  reg = torch.stack(penalties).mean()  # average over collectors for depth-invariance
  
  return reg