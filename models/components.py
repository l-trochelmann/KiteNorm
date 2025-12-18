
import torch
import torch.nn.functional as F
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, dim: int, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

        self.track_variance = False  # While True, keeps a running average over the input variance
        self.register_buffer("running_var_sum", torch.zeros(1))
        self.register_buffer("running_count", torch.zeros(1))
        self.eps = 1e-6

    def forward(self, input):
        mean = input.mean(dim=-1, keepdim=True)
        var = input.var(dim=-1, unbiased=False, keepdim=True)
    
        if self.track_variance:
            current_var = var.float().detach().mean().item()
            self.running_var_sum += current_var
            self.running_count += 1
    
        y = (input - mean) * torch.rsqrt(var + self.eps)
    
        if self.bias is None:
            return y * self.weight
        else:
            return y * self.weight + self.bias


class LayerNorm_Simple(nn.Module):
    """Drops both LN parameters as suggested by Xu et al (2019)"""
    def __init__(self, dim: int):
        super().__init__()
        self.normalized_shape = torch.Size([dim])

    def forward(self, input):
        return F.layer_norm(input, self.normalized_shape, None, None, 1e-6)


class DetachNorm(nn.Module):
    """Detaches mean and variance as shown in experiments by Xu et al (2019)"""
    def __init__(self, dim: int, eps: float = 1e-6, bias: bool = False):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias   = nn.Parameter(torch.zeros(dim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var  = x.var(dim=-1, unbiased=False, keepdim=True)

        mean_detached = mean.detach()
        std_detached  = (var + self.eps).sqrt().detach()

        y = (x - mean_detached) / std_detached

        if self.bias is None:
            y = y * self.weight
        else:
            y = y * self.weight + self.bias
        return y


class OnlyAffine(nn.Module):
    """Ablation: LN gain without normalisation"""
    def __init__(self, dim: int, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

    def forward(self, input):
        output = input * self.weight
        if self.bias is not None:
            output = output + self.bias
        return output


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # x: (bsz, T, dim)
        output = self._norm(x.float()).type_as(x) # (bsz, T, dim)
        return output * self.weight


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int = 256):
        super().__init__()
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.fc1 = nn.Linear(dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        # x: (bsz, T, dim)
        return self.fc2(F.silu(self.fc1(x)))


class GLU(nn.Module):
    """fused GLU"""
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int = 256):
        super().__init__()
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(dim, 2*hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        # x: (bsz, T, dim)
        x, z = self.fc1(x).split(self.hidden_dim, dim=2)
        return self.fc2(F.silu(x) * z)

class MLPReluSquared(nn.Module):
    """MLP with ReLU squared"""
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int = 256):
        super().__init__()
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.fc1 = nn.Linear(dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        # x: (bsz, T, dim)
        return self.fc2(F.relu(self.fc1(x)).pow(2))
