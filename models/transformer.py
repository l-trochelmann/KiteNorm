"""Transformer++, a simple LLama-style Transformer, supporting RMSNorm, RoPE, GLU"""

import math
import torch
import torch.nn.functional as F
from torch import nn
from dataclasses import dataclass

from .components import LayerNorm, LayerNorm_Simple, RMSNorm, MLP, GLU, MLPReluSquared, OnlyAffine, DetachNorm, VarCollector
from .embeddings import precompute_freqs_cis, apply_rotary_emb_complex_like


@dataclass
class ModelConfig:
    vocab_size: int
    seq_len: int
    dim: int
    expand: float
    n_layers: int
    n_heads: int
    weight_init: str
    res_contrib: float
    ln_config: str
    ln_style: str
    attn_style: str
    ln_use_shift: bool = False
    mlp: str = 'mlp'
    rmsorm_eps: float = 1e-6
    tie_embeddings: bool = False
    qknorm_L97: int = 2024
    compile: bool = True
    track_var: bool = False


MLP_CLASSES = {
    "mlp": MLP,
    "glu": GLU,
    "mlp_relu_sq": MLPReluSquared
}


def _get_ln_variant(cfg, dim=None):
    if dim==None:
        dim = cfg.dim
    style = cfg.ln_style.lower()
    if style == 'layernorm':
        return LayerNorm(dim, bias=cfg.ln_use_shift)
    elif style == 'layernorm_simple':
        return LayerNorm_Simple(dim)
    elif style == 'rmsnorm':
        return RMSNorm(dim, cfg.rmsorm_eps)
    elif style == 'onlyaffine':
        return OnlyAffine(dim, bias=cfg.ln_use_shift)
    elif style == 'detachnorm':
        return DetachNorm(dim, bias=cfg.ln_use_shift)
    else:
        raise ValueError("Invalid cfg.ln_style value. Choose from 'LayerNorm', 'LayerNorm_Simple', 'RMSNorm', 'OnlyAffine', 'DetachNorm'")


def _get_attn(cfg):
    style = cfg.attn_style.lower()
    if style == 'default':
        return Attention(cfg)
    elif style == 'qknorm':
        return QKNormAttention(cfg)
    else:
        raise ValueError("Invalid cfg.attn_style value. Choose from 'Default', 'QKNorm'")
        

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

        if cfg.compile:
            self.rope = torch.compile(apply_rotary_emb_complex_like)
        else:
            self.rope = apply_rotary_emb_complex_like
    
    def forward(self, x, freqs_cis):
        bsz, seqlen, d = x.shape # (bsz, seqlen, d)
        
        q, k, v = self.w_qkv(x).split(d, dim=2) # (bsz, seqlen, d)
        q = q.view(bsz, seqlen, self.n_heads, self.head_dim) # (bsz, seqlen, nh, h_dim)
        k = k.view(bsz, seqlen, self.n_heads, self.head_dim) # (bsz, seqlen, nh, h_dim)
        v = v.view(bsz, seqlen, self.n_heads, self.head_dim) # (bsz, seqlen, nh, h_dim)
        
        q, k = self.rope(q, k, freqs_cis=freqs_cis) # (bsz, seqlen, nh, h_dim)
        
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
        
        q, k = self.rope(q, k, freqs_cis=freqs_cis)

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


class Block_NoNorm(nn.Module):
    def __init__(self, layer_id: int, cfg: ModelConfig):
        super().__init__()
        self.attn = _get_attn(cfg)
        self.mlp = MLP_CLASSES[cfg.mlp](dim=cfg.dim, hidden_dim=int(cfg.expand * cfg.dim))
        self.layer_id = layer_id
        self.res_contrib = cfg.res_contrib

        self.track_var = cfg.track_var
        if self.track_var:
            self.coll_attn_in = VarCollector()
            self.coll_attn_out = VarCollector()
            self.coll_attn_add = VarCollector()
            self.coll_mlp_in = VarCollector()
            self.coll_mlp_out = VarCollector()
            self.coll_mlp_add = VarCollector()

    def forward(self, x, freqs_cis):
        # x: (bsz, seqlen, dim)
        if not self.track_var:
            if self.res_contrib == -1:
                x = x + self.attn(x, freqs_cis)
                x = x + self.mlp(x)
            else:
                r = self.res_contrib
                x = math.sqrt(1-r) * x + math.sqrt(r) * self.attn(x, freqs_cis)
                x = math.sqrt(1-r) * x + math.sqrt(r) * self.mlp(x)
            return x
        else:
            x_in = x  # step through attn sublayer
            self.coll_attn_in.calc_var(x_in)
            x_out = self.attn(x, freqs_cis)
            self.coll_attn_out.calc_var(x_out)
            if self.res_contrib == -1:  # no contribution rescaling
                x_add = x_in + x_out
            else:  # rescale contribution to output variance
                r = self.res_contrib
                x_add = math.sqrt(1-r) * x_in + math.sqrt(r) * x_out
            self.coll_attn_add.calc_var(x_add)

            x_in = x_add  # step through mlp sublayer
            self.coll_mlp_in.calc_var(x_in)
            x_out = self.mlp(x_in)
            self.coll_mlp_out.calc_var(x_out)
            if self.res_contrib == -1:  # no contribution rescaling
                x_add = x_in + x_out
            else:  # rescale contribution to output variance
                r = self.res_contrib
                x_add = math.sqrt(1-r) * x_in + math.sqrt(r) * x_out
            self.coll_mlp_add.calc_var(x_add)

            return x_add


class NormBlock(nn.Module):
    def __init__(self, layer_id: int, cfg: ModelConfig):
        super().__init__()
        self.attn = _get_attn(cfg)
        self.attn_norm = _get_ln_variant(cfg)
        self.mlp = MLP_CLASSES[cfg.mlp](dim=cfg.dim, hidden_dim=int(cfg.expand * cfg.dim))
        self.mlp_norm = _get_ln_variant(cfg)
        self.layer_id = layer_id
        self.res_contrib = cfg.res_contrib

        self.track_var = cfg.track_var
        if self.track_var:
            self.coll_attn_in = VarCollector()
            self.coll_attn_out = VarCollector()
            self.coll_attn_add = VarCollector()
            self.coll_mlp_in = VarCollector()
            self.coll_mlp_out = VarCollector()
            self.coll_mlp_add = VarCollector()


class Block_PreNorm(NormBlock):
    def forward(self, x, freqs_cis):
        # x: (bsz, seqlen, dim)
        if not self.track_var:
            if self.res_contrib == -1:
                x = x + self.attn(self.attn_norm(x), freqs_cis)
                x = x + self.mlp(self.mlp_norm(x))
            else:
                r = self.res_contrib
                x = math.sqrt(1-r) * x + math.sqrt(r) * self.attn(self.attn_norm(x), freqs_cis)
                x = math.sqrt(1-r) * x + math.sqrt(r) * self.mlp(self.mlp_norm(x))
            return x
        else:
            x_in = x  # step through attn sublayer
            self.coll_attn_in.calc_var(x_in)
            x_norm = self.attn_norm(x_in)
            x_out = self.attn(x_norm, freqs_cis)
            self.coll_attn_out.calc_var(x_out)
            if self.res_contrib == -1:  # no contribution rescaling
                x_add = x_in + x_out
            else:  # rescale contribution to output variance
                r = self.res_contrib
                x_add = math.sqrt(1-r) * x_in + math.sqrt(r) * x_out
            self.coll_attn_add.calc_var(x_add)

            x_in = x_add  # step through mlp sublayer
            self.coll_mlp_in.calc_var(x_in)
            x_norm = self.mlp_norm(x_in)
            x_out = self.mlp(x_norm)
            self.coll_mlp_out.calc_var(x_out)
            if self.res_contrib == -1:  # no contribution rescaling
                x_add = x_in + x_out
            else:  # rescale contribution to output variance
                r = self.res_contrib
                x_add = math.sqrt(1-r) * x_in + math.sqrt(r) * x_out
            self.coll_mlp_add.calc_var(x_add)

            return x_add


class Block_PostNorm(NormBlock):
    def forward(self, x, freqs_cis):
        # x: (bsz, seqlen, dim)
        if not self.track_var:
            if self.res_contrib == -1:
                x = self.attn_norm(x + self.attn(x, freqs_cis))
                x = self.mlp_norm(x + self.mlp(x))
            else:
                r = self.res_contrib
                x = self.attn_norm(math.sqrt(1-r) * x + math.sqrt(r) * self.attn(x, freqs_cis))
                x = self.mlp_norm(math.sqrt(1-r) * x + math.sqrt(r) * self.mlp(x))

            return x
        else:
            x_in = x  # step through attn sublayer
            self.coll_attn_in.calc_var(x_in)
            x_out = self.attn(x_in, freqs_cis)
            self.coll_attn_out.calc_var(x_out)
            if self.res_contrib == -1:  # no contribution rescaling
                x_add = x_in + x_out
            else:  # rescale contribution to output variance
                r = self.res_contrib
                x_add = math.sqrt(1-r) * x_in + math.sqrt(r) * x_out
            self.coll_attn_add.calc_var(x_add)
            x_norm = self.attn_norm(x_add)

            x_in = x_norm  # step through mlp sublayer
            self.coll_mlp_in.calc_var(x_in)
            x_out = self.mlp(x_in)
            self.coll_mlp_out.calc_var(x_out)
            if self.res_contrib == -1:  # no contribution rescaling
                x_add = x_in + x_out
            else:  # rescale contribution to output variance
                r = self.res_contrib
                x_add = math.sqrt(1-r) * x_in + math.sqrt(r) * x_out
            self.coll_mlp_add.calc_var(x_add)
            x_norm = self.mlp_norm(x_add)

            return x_norm


class Block_PreAttnNorm(nn.Module):
    """ABLATION: In pre-LN, drop MLP norm"""
    def __init__(self, layer_id: int, cfg: ModelConfig):
        super().__init__()
        self.attn = _get_attn(cfg)
        self.attn_norm = _get_ln_variant(cfg)
        self.mlp = MLP_CLASSES[cfg.mlp](dim=cfg.dim, hidden_dim=int(cfg.expand * cfg.dim))
        self.layer_id = layer_id

    def forward(self, x, freqs_cis):
        # x: (bsz, seqlen, dim)
        x = x + self.attn(self.attn_norm(x), freqs_cis)
        x = x + self.mlp(x)
        return x


class Block_PreGLUNorm(nn.Module):
    """ABLATION: In pre-LN, drop Attn norm"""

    def __init__(self, layer_id: int, cfg: ModelConfig):
        super().__init__()
        self.attn = _get_attn(cfg)
        self.mlp = MLP_CLASSES[cfg.mlp](dim=cfg.dim, hidden_dim=int(cfg.expand * cfg.dim))
        self.mlp_norm = _get_ln_variant(cfg)
        self.layer_id = layer_id

    def forward(self, x, freqs_cis):
        # x: (bsz, seqlen, dim)
        x = x + self.attn(x, freqs_cis)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class Block_PostAttnNorm(nn.Module):
    """ABLATION: In post-LN, drop MLP norm"""
    def __init__(self, layer_id: int, cfg: ModelConfig):
        super().__init__()
        self.attn = _get_attn(cfg)
        self.attn_norm = _get_ln_variant(cfg)
        self.mlp = MLP_CLASSES[cfg.mlp](dim=cfg.dim, hidden_dim=int(cfg.expand * cfg.dim))
        self.layer_id = layer_id

    def forward(self, x, freqs_cis):
        # x: (bsz, seqlen, dim)
        x = self.attn_norm(x + self.attn(x, freqs_cis))
        x = x + self.mlp(x)
        return x


class Block_PostGLUNorm(nn.Module):
    def __init__(self, layer_id: int, cfg: ModelConfig):
        super().__init__()
        self.attn = _get_attn(cfg)
        self.mlp = MLP_CLASSES[cfg.mlp](dim=cfg.dim, hidden_dim=int(cfg.expand * cfg.dim))
        self.mlp_norm = _get_ln_variant(cfg)
        self.layer_id = layer_id

    def forward(self, x, freqs_cis):
        # x: (bsz, seqlen, dim)
        x = x + self.attn(x, freqs_cis)
        x = self.mlp_norm(x + self.mlp(x))
        return x


class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.weight_init = cfg.weight_init.lower()
        self.n_layers = cfg.n_layers
        self.dim = cfg.dim
        head_dim = cfg.dim // cfg.n_heads; assert cfg.dim % cfg.n_heads == 0
        
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.out_norm = None
        ln_config = cfg.ln_config.lower()
        if ln_config == 'pre-norm':
            self.layers = nn.ModuleList([Block_PreNorm(idx, cfg) for idx in range(cfg.n_layers)])
            self.out_norm = _get_ln_variant(cfg)
        elif ln_config == 'post-norm':
            self.layers = nn.ModuleList([Block_PostNorm(idx, cfg) for idx in range(cfg.n_layers)])
        elif ln_config == 'pre-attn-norm':  # ABLATION
            self.layers = nn.ModuleList([Block_PreAttnNorm(idx, cfg) for idx in range(cfg.n_layers)])
            self.out_norm = _get_ln_variant(cfg)
        elif ln_config == 'pre-glu-norm':  # ABLATION
            self.layers = nn.ModuleList([Block_PreGLUNorm(idx, cfg) for idx in range(cfg.n_layers)])
            self.out_norm = _get_ln_variant(cfg)
        elif ln_config == 'post-attn-norm':  # ABLATION
            self.layers = nn.ModuleList([Block_PostAttnNorm(idx, cfg) for idx in range(cfg.n_layers)])
        elif ln_config == 'post-glu-norm':  # ABLATION
            self.layers = nn.ModuleList([Block_PostGLUNorm(idx, cfg) for idx in range(cfg.n_layers)])
        elif ln_config == 'no-norm':
            self.layers = nn.ModuleList([Block_NoNorm(idx, cfg) for idx in range(cfg.n_layers)])
        else:
            raise ValueError("Invalid cfg.ln_config value. Choose from 'no-norm', 'pre-norm', 'post-norm', 'pre-attn-norm', 'pre-glu-norm', 'post-attn-norm', 'post-glu-norm'")
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
        if self.out_norm is not None:
            x = self.out_norm(x)
        return self.lm_head(x) # (bsz, seqlen, vocab_size)

    def _init_weights(self, module):
        if self.weight_init in ('gpt2_res-scale', 'gpt2_no-scale', 'gpt2_glu-scale', 'gpt2_attn_scale'):
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        else:
            raise ValueError("Invalid cfg.weight_init value. Choose from 'gpt2_res-scale', 'gpt2_no-scale', 'gpt2_glu-scale', 'gpt2_attn_scale'")

    def _scale_residual_branches(self):
        if self.weight_init in ('gpt2_res-scale'):
            with torch.no_grad():
                for n, p in self.named_parameters():
                    if n.endswith('fc2.weight'): # mlp/glu output layer
                        p.mul_(1/math.sqrt(2 * self.n_layers))
                    if n.endswith('w_out.weight'): # attn output layer
                        p.mul_(1/math.sqrt(2 * self.n_layers))
        elif self.weight_init in ('gpt2_no-scale'):
            return  # no residual scaling at initialisation
        elif self.weight_init == 'gpt2_glu-scale':
            with torch.no_grad():
                for n, p in self.named_parameters():
                    if n.endswith('fc2.weight'): # mlp/glu output layer
                        p.mul_(1/math.sqrt(2 * self.n_layers))
        elif self.weight_init == 'gpt2_attn-scale':
            with torch.no_grad():
                for n, p in self.named_parameters():
                    if n.endswith('w_out.weight'): # attn output layer
                        p.mul_(1/math.sqrt(2 * self.n_layers))
        else:
            raise ValueError("Invalid cfg.weight_init value. Choose from 'gpt2_res-scale', 'gpt2_no-scale', 'gpt2_glu-scale', 'gpt2_attn_scale'")

    def tie_weights(self):
        self.lm_head.weight = self.embed_tokens.weight

    def count_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.embed_tokens.weight.numel()
            if not self.lm_head.weight is self.embed_tokens.weight:  # if no weight tying
                n_params -= self.lm_head.weight.numel()
        return n_params
