
from fractions import Fraction
import wandb


def construct_model(cfg):
  """Initalize a model from config. Counts parameters."""

  # Transformer++
  if cfg.model == "transformer":
    from .transformer import Transformer, ModelConfig
    model_cfg = ModelConfig(
      vocab_size = cfg.vocab_size,
      dim = cfg.d_model,
      expand = float(Fraction(cfg.expand)),
      n_layers = cfg.n_layers,
      n_heads = cfg.n_heads,
      weight_init = cfg.weight_init,
      skip_scale = getattr(cfg, "skip_scale", -1),
      res_scale  = getattr(cfg, "res_scale", -1),
      eps = getattr(cfg, "eps", 1e-6),
      mlp = cfg.mlp_class,
      skip_scale_first_layer = getattr(cfg, "skip_scale_first_layer", -2),
      res_scale_first_layer = getattr(cfg, "res_scale_first_layer", -2),
      omit_outer_norm_first_sublayer = getattr(cfg, "omit_outer_norm_first_sublayer", False),
      seq_len = cfg.seq_len,
      tie_embeddings = cfg.tie_embeddings,
      ln_config = cfg.ln_config,
      ln_style = cfg.ln_style,
      attn_style = getattr(cfg, "attn_style", 'Default'),
      ln_use_shift = getattr(cfg, "ln_use_shift", True),
      qknorm_L97 = getattr(cfg, "qknorm_L97", cfg.seq_len),
      compile = cfg.torch_compile,
      sublayer_tracking = getattr(cfg, "track_sublayer_variance", False)
        or getattr(cfg, "track_sublayer_kurtosis", False)
        or getattr(cfg, "track_token_alignment", False)
        or bool(getattr(cfg, "regulariser", None)),  # For logging  or regulariser
      embedding_norm = getattr(cfg, "embedding_norm", False),
    )
    model = Transformer(model_cfg)
    
  # Pythia
  elif cfg.model.startswith("pythia"):
    from transformers import AutoConfig, AutoModelForCausalLM
    model_cfg = AutoConfig.from_pretrained(f"EleutherAI/{cfg.model}")
    model =  AutoModelForCausalLM.from_config(model_cfg) # NOTE: vocab_size=50304 here!
    model.init_weights() # explict init, since I am not sure it is done in 'from_config'
  
  else:
    raise NotImplementedError(f"Not implemented model: {cfg.model}.")
  
  if hasattr(model, 'count_params'):
    n_params = model.count_params(non_embedding=False)
    n_params_no_embed = model.count_params(non_embedding=True)
    print(f"Number of parameters: {n_params:_}")
    print(f"Number of non-embedding parameters: {n_params_no_embed:_}")
    if wandb.run is not None:
      wandb.log({
        "n_params": n_params,
        "n_params_no_embed": n_params_no_embed
      })
  
  return model, model_cfg


def get_param_groups(model, weight_decay):
  """Create param groups with and withou weight_decay."""
  
  # filter out parameters that do not require grad
  named_param_dict = {n: p for n,p in model.named_parameters() if p.requires_grad}

  # filter out parameters with names containing 'bias', 'norm', etc
  decay_params_names = [n for n, p in model.named_parameters() if not getattr(p, '_no_weight_decay', False)] # exclude mamba 'A_log', 'D'
  decay_params_names = [n for n in decay_params_names if "bias" not in n] # exclude bias
  decay_params_names = [n for n in decay_params_names if "norm" not in n] # exclude normalization layers

  decay_params = [p for n, p in named_param_dict.items() if n in decay_params_names]
  no_decay_params = [p for n, p in named_param_dict.items() if n not in decay_params_names]

  # # sanity check
  # no_decay_param_names = [n for n, p in named_param_dict.items() if n not in decay_params_names]
  # print(f"\nParameters with no weight decay:")
  # print(*no_decay_param_names, sep='\n')
  # print(f"\nParameters with weight decay:")
  # print(*decay_params_names, sep='\n')
  
  param_groups = [
      {"params": decay_params, "weight_decay": weight_decay},
      {"params": no_decay_params, "weight_decay": 0.0},
  ]
  
  return param_groups
