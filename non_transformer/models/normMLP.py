import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass


@dataclass
class MLPConfig:
    n_layers: int
    norm_config: str
    norm_variant: str
    use_res_scale: bool
    use_gain: bool = True
    use_bias: bool = True
    norm_eps: float = 1e-6
    use_residual: bool = True
    use_relu: bool = True


class Norm1d(nn.Module):
    def __init__(self, d_model: int, cfg: MLPConfig):
        super().__init__()
        self.cfg = cfg
        self.weight = nn.Parameter(torch.ones(d_model)) if cfg.use_gain else None
        self.bias = nn.Parameter(torch.zeros(d_model)) if cfg.use_bias else None
        self._norm_shape = (d_model,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variant = self.cfg.norm_variant.lower()
        if variant == 'rmsnorm':
            y = F.rms_norm(x, self._norm_shape, weight=None, eps=self.cfg.norm_eps)
        elif variant == 'layernorm':
            y = F.layer_norm(x, self._norm_shape, weight=None, bias=None, eps=self.cfg.norm_eps)
        else:
            raise ValueError(f"Invalid norm_variant: {self.cfg.norm_variant!r}")
        if self.weight is not None:
            y = y * self.weight
        if self.bias is not None:
            y = y + self.bias
        return y


# ----- Residual blocks -----
class _PreNormResBlock(nn.Module):
    def __init__(self, d_model: int, use_relu: bool, cfg: MLPConfig):
        super().__init__()
        self.norm = Norm1d(d_model, cfg)
        self.fc = nn.Linear(d_model, d_model, bias=False)
        self.use_relu = use_relu

    def forward(self, x):
        y = self.norm(x)
        y = self.fc(y)
        if self.use_relu:
            y = F.relu(y)
        return x + y

class _PostNormResBlock(nn.Module):
    def __init__(self, d_model: int, use_relu: bool, cfg: MLPConfig):
        super().__init__()
        self.fc = nn.Linear(d_model, d_model, bias=False)
        self.norm = Norm1d(d_model, cfg)
        self.use_relu = use_relu

    def forward(self, x):
        y = self.fc(x)
        if self.use_relu:
            y = F.relu(y)
        y = x + y
        return self.norm(y)

class _NoNormResBlock(nn.Module):
    def __init__(self, d_model: int, use_relu: bool, cfg: MLPConfig):
        super().__init__()
        self.fc = nn.Linear(d_model, d_model, bias=False)
        self.use_relu = use_relu

    def forward(self, x):
        y = self.fc(x)
        if self.use_relu:
            y = F.relu(y)
        return x + y

# ----- Non-residual blocks -----
class _NormFFNBlock(nn.Module):
    def __init__(self, d_model: int, use_relu: bool, cfg: MLPConfig):
        super().__init__()
        self.fc = nn.Linear(d_model, d_model, bias=False)
        self.norm = Norm1d(d_model, cfg)
        self.use_relu = use_relu

    def forward(self, x):
        y = self.fc(x)
        if self.use_relu:
            y = F.relu(y)
        return self.norm(y)

class _PureFFNBlock(nn.Module):
    def __init__(self, d_model: int, use_relu: bool, cfg: MLPConfig):
        super().__init__()
        self.fc = nn.Linear(d_model, d_model, bias=False)
        self.use_relu = use_relu

    def forward(self, x):
        y = self.fc(x)
        if self.use_relu:
            y = F.relu(y)
        return y

# ----- Common architecture ----- 
class NormMLP(nn.Module):
    def __init__(self, cfg: MLPConfig, num_classes: int = 10):
        super().__init__()
        self.L = max(cfg.n_layers, 1)
        self.use_relu = cfg.use_relu
        d_model = 512

        self.flatten = nn.Flatten()                                 # 3×32×32 -> 3072
        self.stem = nn.Linear(3 * 32 * 32, d_model, bias=False)     # 3072 -> d_model

        norm_mode = cfg.norm_config.lower()
        is_pre_norm = False
        if cfg.use_residual:
            if norm_mode == "pre-norm":
                block_cls = _PreNormResBlock
                is_pre_norm = True
            elif norm_mode == "post-norm":
                block_cls = _PostNormResBlock
            elif norm_mode == "no-norm":
                block_cls = _NoNormResBlock
            else:
                raise ValueError(f"Unknown norm_config={cfg.norm_config!r} (use 'pre-norm' | 'post-norm' | 'no-norm')")
        else:
            # No residual: treat pre/post as equivalent
            if norm_mode in ("pre-norm", "post-norm"):
                block_cls = _NormFFNBlock
            elif norm_mode == "no-norm":
                block_cls = _PureFFNBlock
            else:
                raise ValueError(f"Unknown norm_config={cfg.norm_config!r} (use 'pre-norm' | 'post-norm' | 'no-norm')")

        self.is_pre_norm = is_pre_norm

        body_blocks = [block_cls(d_model, cfg.use_relu, cfg) for _ in range(self.L)]
        self.body = nn.Sequential(*body_blocks)

        if self.is_pre_norm:
            self.head_norm = Norm1d(d_model, cfg)
        self.head = nn.Linear(d_model, num_classes, bias=False)

        self._init_weights(cfg.use_res_scale)

    def _init_weights(self, use_res_scale: bool):
        # Stem init
        stem_nl = 'relu' if self.use_relu else 'linear'
        nn.init.kaiming_normal_(self.stem.weight, mode='fan_in', nonlinearity=stem_nl)

        # Body init and residual scale
        for b in self.body:
            if hasattr(b, "fc") and isinstance(b.fc, nn.Linear):
                nl = 'relu' if self.use_relu else 'linear'
                nn.init.kaiming_normal_(b.fc.weight, mode='fan_in', nonlinearity=nl)
                if use_res_scale and isinstance(b, (_PreNormResBlock, _PostNormResBlock, _NoNormResBlock)):
                    with torch.no_grad():
                        b.fc.weight.mul_(1.0 / math.sqrt(self.L))

        # Head init
        nn.init.kaiming_normal_(self.head.weight, mode='fan_in', nonlinearity='linear')

    def forward(self, x):
        x = self.flatten(x)
        x = self.stem(x)
        if self.use_relu:
            x = F.relu(x)
        x = self.body(x)
        if self.is_pre_norm:
            x = self.head_norm(x)
        return self.head(x)
