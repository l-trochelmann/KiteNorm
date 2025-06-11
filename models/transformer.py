"""Transformer++, a simple LLama-style Transformer, supporting RMSNorm, RoPE, GLU"""

import math
import torch
import torch.nn.functional as F
from torch import nn
from dataclasses import dataclass

from .components import LayerNorm, LayerNorm_Simple, DyT, RMSNorm, MLP, GLU, MLPReluSquared, GainOnly, DetachNorm, RMSNorm_Simple
from .embeddings import precompute_freqs_cis, apply_rotary_emb_complex_like


@dataclass
class ModelConfig:
    vocab_size: int
    seq_len: int
    dim: int
    expand: float
    n_layers: int
    n_heads: int
    ln_config: str
    ln_style: str
    attn_style: str
    ln_use_shift: bool = False
    dyt_alpha_init: float = 0.5
    mlp: str = 'mlp'
    rmsorm_eps: float = 1e-6
    tie_embeddings: bool = False
    mixLN_ratio: float = 0.25
    qknorm_L97: int = 2024


MLP_CLASSES = {
    "mlp": MLP,
    "glu": GLU,
    "mlp_relu_sq": MLPReluSquared
}


def _get_ln_variant(cfg, dim=None):
    if dim==None:
        dim = cfg.dim

    if cfg.ln_style == 'LayerNorm':
        return LayerNorm(dim, bias=cfg.ln_use_shift)
    elif cfg.ln_style == 'LayerNorm_Simple':
        return LayerNorm_Simple(dim)
    elif cfg.ln_style == 'DyT':
        return DyT(dim, alpha_init_value=cfg.dyt_alpha_init, bias=cfg.ln_use_shift)
    elif cfg.ln_style == 'RMSNorm':
        return RMSNorm(dim, cfg.rmsorm_eps)
    elif cfg.ln_style == 'RMSNorm_Simple':
        return RMSNorm_Simple(dim, cfg.rmsorm_eps)
    elif cfg.ln_style == 'GainOnly':
        return GainOnly(dim, bias=cfg.ln_use_shift)
    elif cfg.ln_style == 'DetachNorm':
        return DetachNorm(dim, bias=cfg.ln_use_shift)
    else:
        raise ValueError("Invalid cfg.ln_style value. Choose from 'LayerNorm', 'LayerNorm_Simple', 'DyT', 'RMSNorm', 'GainOnly', 'DetachNorm', 'RMSNorm_Simple'")


def _get_attn(cfg):
    if cfg.attn_style == 'Default':
        return Attention(cfg)
    elif cfg.attn_style == 'QKNorm':
        return QKNormAttention(cfg)
    elif cfg.attn_style == 'QK-LN':
        return QKLNAttention(cfg)
    elif cfg.attn_style == 'QK-RMSNorm':
        return QKRMSNormAttention(cfg)
    else:
        raise ValueError("Invalid cfg.attn_style value. Choose from 'Default', 'QKNorm', 'QK-LN', 'QK-RMSNorm'")
        


def _scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    """Adapted from docs.pytorch.org, this is a python equivalent to torch.nn.functional.scaled_dot_product_attention"""
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)

    # Calculate total softmax entropy
    token_entropy = -(attn_weight * (attn_weight + 1e-9).log()).sum(dim=-1)  # B × H × L
    sum_entropy = token_entropy.sum()
    n_elems = token_entropy.numel()

    return attn_weight @ value, sum_entropy, n_elems


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.dim % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.dim // cfg.n_heads
        
        self.w_qkv = nn.Linear(cfg.dim, 3*cfg.dim, bias=False)
        self.w_out = nn.Linear(cfg.dim, cfg.dim, bias=False)

        self.track_entropy = False  # While True, all forwards will contribute to a running average of softmax entropy
        self.register_buffer("entropy_sum", torch.zeros(1))
        self.register_buffer("entropy_count", torch.zeros(1))
    
    def forward(self, x, freqs_cis):
        bsz, seqlen, d = x.shape # (bsz, seqlen, d)
        
        q, k, v = self.w_qkv(x).split(d, dim=2) # (bsz, seqlen, d)
        q = q.view(bsz, seqlen, self.n_heads, self.head_dim) # (bsz, seqlen, nh, h_dim)
        k = k.view(bsz, seqlen, self.n_heads, self.head_dim) # (bsz, seqlen, nh, h_dim)
        v = v.view(bsz, seqlen, self.n_heads, self.head_dim) # (bsz, seqlen, nh, h_dim)
        
        q, k = apply_rotary_emb_complex_like(q, k, freqs_cis=freqs_cis) # (bsz, seqlen, nh, h_dim)
        
        q = q.transpose(1, 2) # (bsz, nh, seqlen, h_dim)
        k = k.transpose(1, 2) # (bsz, nh, seqlen, h_dim)
        v = v.transpose(1, 2) # (bsz, nh, seqlen, h_dim)

        if not self.track_entropy:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True) # (bsz, nh, seqlen, h_dim)
        else:
            out, sum_ent, n = _scaled_dot_product_attention(q, k, v, is_causal=True)
            self.entropy_sum.add_(sum_ent.detach())
            self.entropy_count.add_(n)
        
        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, d) # (bsz, seqlen, d)
        
        return self.w_out(out)


class QKNormAttention(Attention):
    """QKNorm Attention following Henry et al (2020)"""
    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)

        g0 = math.log2(cfg.qknorm_L97 * cfg.qknorm_L97 - cfg.qknorm_L97)  # Initialise temperature
        self.g = nn.Parameter(torch.tensor(g0))


    def forward(self, x, freqs_cis):
        bsz, seqlen, d = x.shape
        
        q, k, v = self.w_qkv(x).split(d, dim=2)
        q = q.view(bsz, seqlen, self.n_heads, self.head_dim)
        k = k.view(bsz, seqlen, self.n_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_heads, self.head_dim)
        
        q, k = apply_rotary_emb_complex_like(q, k, freqs_cis=freqs_cis)

        q = q / (q.norm(dim=-1, keepdim=True))  # l2 normalisation across the head dimension
        k = k / (k.norm(dim=-1, keepdim=True))  # l2 normalisation across the head dimension

        q = q * self.g  # Apply temperature. Note that (q*g)k^T =  g*(qk^T)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if not self.track_entropy:
            out = F.scaled_dot_product_attention(q, k, v, scale=1, is_causal=True)  # temperature is already applied, deactivate default 1/sqrt(d) scaling
        else:
            out, sum_ent, n = _scaled_dot_product_attention(q, k, v, scale=1, is_causal=True)   # temperature is already applied, deactivate default 1/sqrt(d) scaling
            self.entropy_sum.add_(sum_ent.detach())
            self.entropy_count.add_(n)
        
        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, d)

        return self.w_out(out)
    

class QKLNAttention(Attention):
    """QKNorm Attention, but using LN on the queries and keys, and no scalar temperature"""
    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)

        self.qk_norm = LayerNorm(dim=self.head_dim, bias=cfg.ln_use_shift)  # LN across the head dimension


    def forward(self, x, freqs_cis):
        bsz, seqlen, d = x.shape
        
        q, k, v = self.w_qkv(x).split(d, dim=2)
        q = q.view(bsz, seqlen, self.n_heads, self.head_dim)
        k = k.view(bsz, seqlen, self.n_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_heads, self.head_dim)
        
        q, k = apply_rotary_emb_complex_like(q, k, freqs_cis=freqs_cis)

        q = self.qk_norm(q)  # LN instead of l2 norm
        k = self.qk_norm(k)  # LN instead of l2 norm
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if not self.track_entropy:
            out = F.scaled_dot_product_attention(q, k, v, scale=1/self.head_dim, is_causal=True)  # use fixed 1/d scale, no temperature
        else:
            out, sum_ent, n = _scaled_dot_product_attention(q, k, v, scale=1/self.head_dim, is_causal=True)   # use fixed 1/d scale, no temperature
            self.entropy_sum.add_(sum_ent.detach())
            self.entropy_count.add_(n)
        
        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, d)

        return self.w_out(out)

class QKRMSNormAttention(Attention):
    """QKNorm Attention, but using RMSNorm on the queries and keys, and no scalar temperature"""
    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)

        self.qk_norm = RMSNorm(dim=self.head_dim)  # RMSNorm across the head dimension

        g0 = math.log2(cfg.qknorm_L97 * cfg.qknorm_L97 - cfg.qknorm_L97)  # ABLATION: QKRMSNorm with QKNorm temperature init
        g0 = math.sqrt(g0)  # Note a*(u dot v) = (sqrt(a)*u) dot (sqrt(a)*v)
        with torch.no_grad():
            self.qk_norm.weight.data.mul_(g0)

    def forward(self, x, freqs_cis):
        bsz, seqlen, d = x.shape
        
        q, k, v = self.w_qkv(x).split(d, dim=2)
        q = q.view(bsz, seqlen, self.n_heads, self.head_dim)
        k = k.view(bsz, seqlen, self.n_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_heads, self.head_dim)
        
        q, k = apply_rotary_emb_complex_like(q, k, freqs_cis=freqs_cis)

        q = self.qk_norm(q)  # RMSNorm instead of l2 norm
        k = self.qk_norm(k)  # RMSNorm instead of l2 norm
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if not self.track_entropy:
            out = F.scaled_dot_product_attention(q, k, v, scale=1/self.head_dim, is_causal=True)  # use fixed 1/d scale, no scalar temperature
        else:
            out, sum_ent, n = _scaled_dot_product_attention(q, k, v, scale=1/self.head_dim, is_causal=True)   # use fixed 1/d scale, no scalar temperature
            self.entropy_sum.add_(sum_ent.detach())
            self.entropy_count.add_(n)
        
        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, d)

        return self.w_out(out)


class Block_NoLN(nn.Module):
    def __init__(self, layer_id: int, cfg: ModelConfig):
        super().__init__()
        self.attn = _get_attn(cfg)
        self.mlp = MLP_CLASSES[cfg.mlp](dim=cfg.dim, hidden_dim=int(cfg.expand * cfg.dim))
        self.layer_id = layer_id

    def forward(self, x, freqs_cis):
        # x: (bsz, seqlen, dim)
        x = x + self.attn(x, freqs_cis)
        x = x + self.mlp(x)
        return x 


class Block_ReZero(nn.Module):
    def __init__(self, layer_id: int, cfg: ModelConfig):
        super().__init__()
        self.attn = _get_attn(cfg)
        self.mlp = MLP_CLASSES[cfg.mlp](dim=cfg.dim, hidden_dim=int(cfg.expand * cfg.dim))
        self.attn_resweight = nn.Parameter(torch.Tensor([0]))
        self.mlp_resweight = nn.Parameter(torch.Tensor([0]))
        self.layer_id = layer_id

    def forward(self, x, freqs_cis):
        # x: (bsz, seqlen, dim)
        x = x + self.attn_resweight * self.attn(x, freqs_cis)
        x = x + self.mlp_resweight * self.mlp(x)
        return x    


class Block_LN(nn.Module):
    def __init__(self, layer_id: int, cfg: ModelConfig):
        super().__init__()
        self.attn = _get_attn(cfg)
        self.attn_norm = _get_ln_variant(cfg)
        self.mlp = MLP_CLASSES[cfg.mlp](dim=cfg.dim, hidden_dim=int(cfg.expand * cfg.dim))
        self.mlp_norm = _get_ln_variant(cfg)
        self.layer_id = layer_id


class Block_PreLN(Block_LN):
    def forward(self, x, freqs_cis):
        # x: (bsz, seqlen, dim)
        x = x + self.attn(self.attn_norm(x), freqs_cis)
        x = x + self.mlp(self.mlp_norm(x))
        return x
    

class Block_PostLN(Block_LN):
    def forward(self, x, freqs_cis):
        # x: (bsz, seqlen, dim)
        x = self.attn_norm(x + self.attn(x, freqs_cis))
        x = self.mlp_norm(x + self.mlp(x))
        return x


class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_layers = cfg.n_layers
        head_dim = cfg.dim // cfg.n_heads; assert cfg.dim % cfg.n_heads == 0
        
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.ln_config = cfg.ln_config
        if cfg.ln_config == 'pre-LN':  # Use pre-LN blocks and out_norm
            self.layers = nn.ModuleList([Block_PreLN(idx, cfg) for idx in range(cfg.n_layers)])
            self.out_norm = _get_ln_variant(cfg)
        elif cfg.ln_config == 'post-LN':  # Use post-LN blocks
            self.layers = nn.ModuleList([Block_PostLN(idx, cfg) for idx in range(cfg.n_layers)])
        elif cfg.ln_config == 'mix-LN':  # Use partly post-LN and partly pre-LN based on a ratio parameter, followed by out_norm. Post-LN first.
            self.layers = nn.ModuleList([Block_PostLN(idx, cfg) if idx < math.floor(cfg.mixLN_ratio * cfg.n_layers)
                else Block_PreLN(idx, cfg)
                for idx in range(cfg.n_layers)])
            self.out_norm = _get_ln_variant(cfg)
        elif cfg.ln_config == 'ReZero':  # Use ReZero blocks without any norm
            self.layers = nn.ModuleList([Block_ReZero(idx, cfg) for idx in range(cfg.n_layers)])
        elif cfg.ln_config == 'None':
            self.layers = nn.ModuleList([Block_NoLN(idx, cfg) for idx in range(cfg.n_layers)])
        else:
            raise ValueError("Invalid cfg.ln_config value. Choose from 'pre-LN', 'post-LN', 'mix-LN', 'ReZero', 'None'")
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        
        self.freqs_cis = precompute_freqs_cis(head_dim, cfg.seq_len, 500000)[0:cfg.seq_len]
        
        # init all weights, scale residual branches
        self.apply(self._init_weights)
        self._scale_residual_branches()
        
        if cfg.tie_embeddings:
            self.tie_weights()

    def forward(self, x):
        # x: (bsz, seqlen)
        x = self.embed_tokens(x) # (bsz, seqlen, dim)
        self.freqs_cis = self.freqs_cis.to(x.device)
        for layer in self.layers:
            x = layer(x, self.freqs_cis) # (bsz, seqlen, dim)
        if self.ln_config in ('pre-LN', 'mix-LN'):  # Only use out_norm when needed
            x = self.out_norm(x)
        return self.lm_head(x) # (bsz, seqlen, vocab_size)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _scale_residual_branches(self):
        for n, p in self.named_parameters():
            if n.endswith('fc2.weight'): # mlp/glu output layer
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * self.n_layers))
            if n.endswith('w_out.weight'): # attn output layer
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * self.n_layers))

    def tie_weights(self):
        self.lm_head.weight = self.embed_tokens.weight

    def count_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.embed_tokens.weight.numel()
            if not self.lm_head.weight is self.embed_tokens.weight:  # if no weight tying
                n_params -= self.lm_head.weight.numel()
        return n_params
