import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from dataclasses import dataclass


@dataclass
class CNNConfig:
    n_blocks: int
    norm_config: str
    norm_variant: str
    use_res_scale: bool
    use_gain: bool = True
    use_bias: bool = True
    norm_eps: float = 1e-6


class Norm2d(nn.Module):
    """
    Layer/RMS norm for NCHW tensors with statistics over (C,H,W).
    Uses F.layer_norm / F.rms_norm without affine, then applies
    per-channel affine to mimic GroupNorm(1, C).
    """
    def __init__(self, num_channels: int, cfg: CNNConfig):
        super().__init__()
        self.cfg = cfg
        self.weight = nn.Parameter(torch.ones(num_channels)) if cfg.use_gain else None
        self.bias = nn.Parameter(torch.zeros(num_channels)) if cfg.use_bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.shape
        y = x.view(N, -1)  # normalize over all (C*H*W) features per sample
        norm_shape = (y.shape[-1],)

        variant = self.cfg.norm_variant.lower()
        if variant == 'rmsnorm':
            y = F.rms_norm(y, norm_shape, weight=None, eps=self.cfg.norm_eps)
        elif variant == 'layernorm':
            y = F.layer_norm(y, norm_shape, weight=None, bias=None, eps=self.cfg.norm_eps)
        else:
            raise ValueError(f"Invalid norm_variant: {self.cfg.norm_variant!r}")
        
        y = y.view_as(x)

        if self.weight is not None:
            w = self.weight.view(1, -1, 1, 1)
            y = y * w
        if self.bias is not None:
            b = self.bias.view(1, -1, 1, 1)
            y = y + b

        return y


# Blocks
class NormBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=True)
        else:
            self.shortcut = nn.Identity()

class PreNormBlock(NormBlock):
    def __init__(self, in_planes, planes, stride=1, cfg: CNNConfig = None):
        super().__init__(in_planes, planes, stride)
        self.norm = Norm2d(in_planes, cfg)

    def forward(self, x):
        out = self.norm(x)
        out = F.relu(self.conv1(out))
        out = out + self.shortcut(x)
        return out

class PostNormBlock(NormBlock):
    def __init__(self, in_planes, planes, stride=1, cfg: CNNConfig = None):
        super().__init__(in_planes, planes, stride)
        self.norm = Norm2d(planes, cfg)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = out + self.shortcut(x)
        out = self.norm(out)
        return out

class NoNormBlock(NormBlock):
    def __init__(self, in_planes, planes, stride=1, cfg: CNNConfig = None):
        super().__init__(in_planes, planes, stride)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = out + self.shortcut(x)
        return out


# Architecture
class NormCNN(nn.Module):
    def __init__(self, cfg: CNNConfig):
        super().__init__()
        self.in_planes = 64

        norm_mode = cfg.norm_config.lower()
        if norm_mode == "pre-norm":
            block_cls = PreNormBlock
            is_pre_norm = True
        elif norm_mode == "post-norm":
            block_cls = PostNormBlock
            is_pre_norm = False
        elif norm_mode == "no-norm":
            block_cls = NoNormBlock
            is_pre_norm = False
        else:
            raise ValueError(f"Unknown norm_config={cfg.norm_config!r} (use 'pre-norm' | 'post-norm' | 'no-norm')")

        self.is_pre_norm = is_pre_norm
        self.use_res_scale = cfg.use_res_scale

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        n = cfg.n_blocks
        self.layer1 = self._make_layer(block_cls, 64,  n, stride=1,  cfg=cfg)
        self.layer2 = self._make_layer(block_cls, 128, n, stride=2,  cfg=cfg)
        self.layer3 = self._make_layer(block_cls, 256, n, stride=2,  cfg=cfg)
        self.layer4 = self._make_layer(block_cls, 512, n, stride=2,  cfg=cfg)
        if self.is_pre_norm:
            self.head_norm = Norm2d(512, cfg)
        self.head = nn.Linear(512*block_cls.expansion, 10, bias=False)

    def _make_layer(self, block, planes, num_blocks, stride, cfg: CNNConfig):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        if self.use_res_scale:  # yields cumulative variance that is invariant to n_blocks. Derived for the conv layer specifically.
            res_scale = math.sqrt(2.0) / math.sqrt(num_blocks)  
        for stride in strides:
            b = block(self.in_planes, planes, stride, cfg=cfg)
            if self.use_res_scale:
                with torch.no_grad():
                    b.conv1.weight.mul_(res_scale)
            layers.append(b)
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        if self.is_pre_norm:
            out = self.head_norm(out)
        out = out.view(out.size(0), -1)
        out = self.head(out)
        return out
