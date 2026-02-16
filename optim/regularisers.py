"""Custom regularisers for improving/replacing LayerNorm"""

import torch

def sum_of_variances(model):
    vars_ = []
    for m in model.modules():
      if type(m).__name__ in {"LayerNorm"}:
        v = getattr(m, "last_var", None)
        if v is not None:
          vars_.append(v)
    if not vars_:
      raise NotImplementedError("Attempted to fetch normalisation variances for the regulariser, but model has no normalisation layers!")
    vars = torch.stack(vars_)  # shape: (n_norm_layers,)
    reg = vars.sum()  # replace
    return reg

