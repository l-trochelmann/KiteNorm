import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----- Residual blocks -----
class _PreNormResBlock(nn.Module):
    def __init__(self, d_model: int, use_relu: bool):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, d_model, bias=False)
        self.use_relu = use_relu

    def forward(self, x):
        y = self.norm(x)
        y = self.fc(y)
        if self.use_relu:
            y = F.relu(y)
        return x + y

class _PostNormResBlock(nn.Module):
    def __init__(self, d_model: int, use_relu: bool):
        super().__init__()
        self.fc = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.use_relu = use_relu

    def forward(self, x):
        y = self.fc(x)
        if self.use_relu:
            y = F.relu(y)
        y = x + y
        return self.norm(y)

class _NoNormResBlock(nn.Module):
    def __init__(self, d_model: int, use_relu: bool):
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
    def __init__(self, d_model: int, use_relu: bool):
        super().__init__()
        self.fc = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.use_relu = use_relu

    def forward(self, x):
        y = self.fc(x)
        if self.use_relu:
            y = F.relu(y)
        return self.norm(y)

class _PureFFNBlock(nn.Module):
    def __init__(self, d_model: int, use_relu: bool):
        super().__init__()
        self.fc = nn.Linear(d_model, d_model, bias=False)
        self.use_relu = use_relu

    def forward(self, x):
        y = self.fc(x)
        if self.use_relu:
            y = F.relu(y)
        return y

# ----- Common architecture ----- 
class _NormMLP(nn.Module):
    def __init__(self, block_cls, L: int, use_relu: bool,
                 d_model: int = 512, num_classes: int = 10, use_res_scale: bool = False):
        super().__init__()
        self.L = max(L, 1)
        self.use_relu = use_relu
        self.flatten = nn.Flatten()                              # 3×32×32 -> 3072
        self.stem = nn.Linear(3 * 32 * 32, d_model, bias=False)  # 3072 -> d_model

        body_blocks = [block_cls(d_model, use_relu) for _ in range(L)]
        self.body = nn.Sequential(*body_blocks)
        self.head = nn.Linear(d_model, num_classes)              # d_model -> 10

        self._init_weights(use_res_scale)

    def _init_weights(self, use_res_scale: bool):
        # Stem: followed by ReLU
        stem_nl = 'relu' if self.use_relu else 'linear'
        nn.init.kaiming_normal_(self.stem.weight, mode='fan_in', nonlinearity=stem_nl)

        # Body blocks
        for b in self.body:
            if hasattr(b, "fc") and isinstance(b.fc, nn.Linear):
                nl = 'relu' if self.use_relu else 'linear'
                nn.init.kaiming_normal_(b.fc.weight, mode='fan_in', nonlinearity=nl)
                # residual scaling for residual variants only
                if use_res_scale:
                    with torch.no_grad():
                        b.fc.weight.mul_(1.0 / math.sqrt(self.L))

        # Head: linear (no activation after)
        nn.init.kaiming_normal_(self.head.weight, mode='fan_in', nonlinearity='linear')
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        x = self.flatten(x)
        x = self.stem(x)
        if self.use_relu:
            x = F.relu(x)
        x = self.body(x)
        return self.head(x)

# ----- Public constructors -----
def PreNormResidual(L: int, use_relu: bool, d_model: int = 512,
                    num_classes: int = 10, use_res_scale: bool = False) -> nn.Module:
    return _NormMLP(_PreNormResBlock, L, use_relu, d_model, num_classes, use_res_scale)

def PostNormResidual(L: int, use_relu: bool, d_model: int = 512,
                     num_classes: int = 10, use_res_scale: bool = False) -> nn.Module:
    return _NormMLP(_PostNormResBlock, L, use_relu, d_model, num_classes, use_res_scale)

def NoNormResidual(L: int, use_relu: bool, d_model: int = 512,
                   num_classes: int = 10, use_res_scale: bool = False) -> nn.Module:
    return _NormMLP(_NoNormResBlock, L, use_relu, d_model, num_classes, use_res_scale)

def NormFFN(L: int, use_relu: bool, d_model: int = 512, num_classes: int = 10) -> nn.Module:
    return _NormMLP(_NormFFNBlock, L, use_relu, d_model, num_classes, use_res_scale=False)

def PureFFN(L: int, use_relu: bool, d_model: int = 512, num_classes: int = 10) -> nn.Module:
    return _NormMLP(_PureFFNBlock, L, use_relu, d_model, num_classes, use_res_scale=False)
