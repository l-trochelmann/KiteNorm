"""Custom regularisers for improving/replacing LayerNorm"""

import torch
import torch.nn.functional as F


def var_ReLU(model):
  """
  Mean of ReLU(X-1) where X is variance
  """
  which_collectors = (
    "coll_attn_add",
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

  reg = torch.stack(penalties).mean()

  return reg
