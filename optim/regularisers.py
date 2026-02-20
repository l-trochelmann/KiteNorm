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
  Mean of L2 distances between current variances and target variances
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
        penalty = torch.square(coll.last_var - 1.0).mean()  # average tokenwise L2 distance
        penalties.append(penalty)

  if not penalties:
    raise NotImplementedError("Attempting to fetch sublayer variances for the regulariser, but no specified VarCollector was found.")
  
  reg = torch.stack(penalties).mean()  # average over collectors for depth-invariance
  
  return reg


def mean_LogLin(model):
  """
  Mean of X - LOG(X) - 1 penalties where X is variance
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
        v = coll.last_var
        penalty = (v - torch.log(v) - 1).mean()  # average tokenwise log-linear penalty
        penalties.append(penalty)

  if not penalties:
    raise NotImplementedError("Attempting to fetch sublayer variances for the regulariser, but no specified VarCollector was found.")
  
  reg = torch.stack(penalties).mean()  # average over collectors for depth-invariance
  
  return reg


def mean_LogSqr(model):
  """
  Mean of X^2 - LOG(X^2) - 1 penalties where X is variance
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
        v_sqr = torch.square(coll.last_var)
        penalty = (v_sqr - torch.log(v_sqr) - 1).mean()  # average tokenwise log-square penalty
        penalties.append(penalty)

  if not penalties:
    raise NotImplementedError("Attempting to fetch sublayer variances for the regulariser, but no specified VarCollector was found.")
  
  reg = torch.stack(penalties).mean()  # average over collectors for depth-invariance
  
  return reg