"""Transformer++, a simple LLama-style Transformer, supporting RMSNorm, RoPE, GLU"""

import math
import torch
import torch.nn.functional as F
from torch import nn
from dataclasses import dataclass

from .components import LayerNorm, LayerNorm_Scaling, ScalarLN, RMSNorm, ScalarRMSNorm, MLP, GLU, MLPReluSquared, VarCollector
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
    skip_scale: float
    res_scale: float
    ln_config: str
    ln_style: str
    ln_use_shift: bool = False
    mlp: str = 'mlp'
    skip_scale_first_layer: float = -2
    res_scale_first_layer: float = -2
    omit_outer_norm_first_sublayer: bool = False
    eps: float = 1e-6
    tie_embeddings: bool = False
    compile: bool = True
    sublayer_tracking: bool = False
    mixNorm_ratio: float = 0.25


MLP_CLASSES = {
    "mlp": MLP,
    "glu": GLU,
    "mlp_relu_sq": MLPReluSquared
}


def _get_ln_variant(cfg, dim=None, layer_idx=None):
    if dim is None:
        dim = cfg.dim
    style = cfg.ln_style.lower()
    if style == "layernorm":
        return LayerNorm(dim, eps=cfg.eps, bias=cfg.ln_use_shift)
    elif style == "layernorm_scaling":
        return LayerNorm_Scaling(dim, layer_idx=layer_idx, eps=cfg.eps, bias=cfg.ln_use_shift) if layer_idx != -1 else LayerNorm(dim, eps=cfg.eps, bias=cfg.ln_use_shift)
    elif style == "rmsnorm":
        return RMSNorm(dim, cfg.eps)
    elif style == "scalarln":
        return ScalarLN(eps=cfg.eps, bias=cfg.ln_use_shift)
    elif style == "scalarrmsnorm":
        return ScalarRMSNorm(eps=cfg.eps)
    else:
        raise ValueError("Invalid cfg.ln_style value. Choose from 'LayerNorm', 'LayerNorm_Scaling', 'ScalarLN', 'RMSNorm', 'ScalarRMSNorm'")


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.dim % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.dim // cfg.n_heads

        self.w_qkv = nn.Linear(cfg.dim, 3*cfg.dim, bias=False)
        self.w_out = nn.Linear(cfg.dim, cfg.dim, bias=False)

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

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True) # (bsz, nh, seqlen, h_dim)

        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, d) # (bsz, seqlen, d)

        return self.w_out(out)


class NormBlock(nn.Module):
    def __init__(self, layer_id: int, cfg: ModelConfig):
        super().__init__()
        self.attn = Attention(cfg)
        self.attn_norm = _get_ln_variant(cfg, layer_idx=layer_id)
        self.mlp = MLP_CLASSES[cfg.mlp](dim=cfg.dim, hidden_dim=int(cfg.expand * cfg.dim))
        self.mlp_norm = _get_ln_variant(cfg, layer_idx=layer_id)
        self.layer_id = layer_id

        self.skip_scale = cfg.skip_scale
        self.res_scale = cfg.res_scale

        self.sublayer_tracking = cfg.sublayer_tracking
        if self.sublayer_tracking:
            self.coll_attn_in  = VarCollector()
            self.coll_attn_out = VarCollector()
            self.coll_attn_add = VarCollector(is_after_add=True)
            self.coll_mlp_in   = VarCollector()
            self.coll_mlp_out  = VarCollector()
            self.coll_mlp_add  = VarCollector(is_after_add=True)

    def _mix(self, x_in, x_out):
        if self.skip_scale != -1:
            x_in = self.skip_scale * x_in
        if self.res_scale != -1:
            x_out = self.res_scale * x_out
        return x_in + x_out


class Block_PreNorm(NormBlock):
    def forward(self, x, freqs_cis):
        # x: (bsz, seqlen, dim)
        if not self.sublayer_tracking:
            x = self._mix(x, self.attn(self.attn_norm(x), freqs_cis))
            x = self._mix(x, self.mlp(self.mlp_norm(x)))
            return x

        # attn
        x_in = x
        self.coll_attn_in.calc_var(x_in)
        x_out = self.attn(self.attn_norm(x_in), freqs_cis)
        self.coll_attn_out.calc_var(x_out)
        x_add = self._mix(x_in, x_out)
        self.coll_attn_add.calc_var(x_add)

        # mlp
        x_in = x_add
        self.coll_mlp_in.calc_var(x_in)
        x_out = self.mlp(self.mlp_norm(x_in))
        self.coll_mlp_out.calc_var(x_out)
        x_add = self._mix(x_in, x_out)
        self.coll_mlp_add.calc_var(x_add)

        return x_add


class Block_PostNorm(NormBlock):
    def __init__(self, layer_id: int, cfg: ModelConfig):
        super().__init__(layer_id, cfg)
        self.skip_scale_first_layer = cfg.skip_scale_first_layer
        self.res_scale_first_layer = cfg.res_scale_first_layer
        self.omit_outer_norm_first_sublayer = cfg.omit_outer_norm_first_sublayer

        if self.omit_outer_norm_first_sublayer and self.layer_id == 0:
            self.attn_norm = None

    def _scales(self):
        s_skip, s_res = self.skip_scale, self.res_scale
        if self.layer_id == 0:
            if self.skip_scale_first_layer != -2:
                s_skip = self.skip_scale_first_layer
            if self.res_scale_first_layer != -2:
                s_res = self.res_scale_first_layer
        return s_skip, s_res

    def _mix2(self, x_in, x_out, skip_scale, res_scale):
        if skip_scale != -1:
            x_in = skip_scale * x_in
        if res_scale != -1:
            x_out = res_scale * x_out
        return x_in + x_out

    def forward(self, x, freqs_cis):
        skip_scale, res_scale = self._scales()

        if not self.sublayer_tracking:
            # attn sublayer
            x_add = self._mix2(x, self.attn(x, freqs_cis), skip_scale, res_scale)
            if self.attn_norm is None:
                x = x_add
            else:
                x = self.attn_norm(x_add)

            # mlp sublayer
            x_add = self._mix2(x, self.mlp(x), skip_scale, res_scale)
            x = self.mlp_norm(x_add)
            return x

        # attn
        x_in = x
        self.coll_attn_in.calc_var(x_in)
        x_out = self.attn(x_in, freqs_cis)
        self.coll_attn_out.calc_var(x_out)
        x_add = self._mix2(x_in, x_out, skip_scale, res_scale)
        self.coll_attn_add.calc_var(x_add)

        if self.attn_norm is None:
            x_norm = x_add
        else:
            x_norm = self.attn_norm(x_add)

        # mlp
        x_in = x_norm
        self.coll_mlp_in.calc_var(x_in)
        x_out = self.mlp(x_in)
        self.coll_mlp_out.calc_var(x_out)
        x_add = self._mix2(x_in, x_out, skip_scale, res_scale)
        self.coll_mlp_add.calc_var(x_add)
        x_norm = self.mlp_norm(x_add)

        return x_norm


class Block_DoubleNorm(nn.Module):
    """ pre- and post-normalisation as proposed in https://arxiv.org/pdf/2601.19895 """
    def __init__(self, layer_id: int, cfg: ModelConfig):
        super().__init__()
        self.layer_id = layer_id

        self.attn = Attention(cfg)
        self.mlp = MLP_CLASSES[cfg.mlp](dim=cfg.dim, hidden_dim=int(cfg.expand * cfg.dim))

        # two norms per sublayer
        self.attn_norm_in  = _get_ln_variant(cfg, layer_idx=layer_id)
        self.attn_norm_out = None if (cfg.omit_outer_norm_first_sublayer and layer_id == 0) else _get_ln_variant(cfg, layer_idx=layer_id)
        self.mlp_norm_in   = _get_ln_variant(cfg, layer_idx=layer_id)
        self.mlp_norm_out  = _get_ln_variant(cfg, layer_idx=layer_id)

        self.skip_scale = cfg.skip_scale
        self.res_scale = cfg.res_scale

        # config-driven exceptions
        self.skip_scale_first_layer = cfg.skip_scale_first_layer
        self.res_scale_first_layer = cfg.res_scale_first_layer 
        self.omit_outer_norm_first_sublayer = cfg.omit_outer_norm_first_sublayer

        self.sublayer_tracking = cfg.sublayer_tracking
        if self.sublayer_tracking:
            self.coll_attn_in  = VarCollector()
            self.coll_attn_out = VarCollector()
            self.coll_attn_add = VarCollector(is_after_add=True)
            self.coll_mlp_in   = VarCollector()
            self.coll_mlp_out  = VarCollector()
            self.coll_mlp_add  = VarCollector(is_after_add=True)

    def _scales(self):
        """Return (skip_scale, res_scale) after applying layer-0 overrides if configured."""
        s_skip, s_res = self.skip_scale, self.res_scale
        if self.layer_id == 0:
            if self.skip_scale_first_layer != -2:
                s_skip = self.skip_scale_first_layer
            if self.res_scale_first_layer != -2:
                s_res = self.res_scale_first_layer
        return s_skip, s_res

    def _mix(self, x_in, x_out, skip_scale, res_scale):
        if skip_scale != -1:
            x_in = skip_scale * x_in
        if res_scale != -1:
            x_out = res_scale * x_out
        return x_in + x_out

    def forward(self, x, freqs_cis):
        skip_scale, res_scale = self._scales()

        # attn sublayer
        if self.sublayer_tracking:
            x_in = x
            self.coll_attn_in.calc_var(x_in)
            x_out = self.attn(self.attn_norm_in(x_in), freqs_cis)
            self.coll_attn_out.calc_var(x_out)
            x_add = self._mix(x_in, x_out, skip_scale, res_scale)
            self.coll_attn_add.calc_var(x_add)
        else:
            x_add = self._mix(x, self.attn(self.attn_norm_in(x), freqs_cis), skip_scale, res_scale)

        if self.attn_norm_out is None:
            x = x_add
        else:
            x = self.attn_norm_out(x_add)

        # mlp sublayer
        if self.sublayer_tracking:
            x_in = x
            self.coll_mlp_in.calc_var(x_in)
            x_out = self.mlp(self.mlp_norm_in(x_in))
            self.coll_mlp_out.calc_var(x_out)
            x_add = self._mix(x_in, x_out, skip_scale, res_scale)
            self.coll_mlp_add.calc_var(x_add)
            x = self.mlp_norm_out(x_add)
        else:
            x = self.mlp_norm_out(self._mix(x, self.mlp(self.mlp_norm_in(x)), skip_scale, res_scale))

        return x


class Block_PeriNorm(Block_DoubleNorm):
    """Peri-LN aka Sandwich-LN as discussed in https://arxiv.org/pdf/2502.02732."""
    def forward(self, x, freqs_cis):
        skip_scale, res_scale = self._scales()

        # attn sublayer
        if self.sublayer_tracking:
            x_in = x
            self.coll_attn_in.calc_var(x_in)
            x_out = self.attn(self.attn_norm_in(x_in), freqs_cis)
            if self.attn_norm_out is not None:
                x_out = self.attn_norm_out(x_out)
            self.coll_attn_out.calc_var(x_out)
            x = self._mix(x_in, x_out, skip_scale, res_scale)
            self.coll_attn_add.calc_var(x)
        else:
            x_out = self.attn(self.attn_norm_in(x), freqs_cis)
            if self.attn_norm_out is not None:
                x_out = self.attn_norm_out(x_out)
            x = self._mix(x, x_out, skip_scale, res_scale)

        # mlp sublayer
        if self.sublayer_tracking:
            x_in = x
            self.coll_mlp_in.calc_var(x_in)
            x_out = self.mlp(self.mlp_norm_in(x_in))
            x_out = self.mlp_norm_out(x_out)
            self.coll_mlp_out.calc_var(x_out)
            x = self._mix(x_in, x_out, skip_scale, res_scale)
            self.coll_mlp_add.calc_var(x)
        else:
            x_out = self.mlp(self.mlp_norm_in(x))
            x_out = self.mlp_norm_out(x_out)
            x = self._mix(x, x_out, skip_scale, res_scale)

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
            self.out_norm = _get_ln_variant(cfg, layer_idx=-1)
        elif ln_config == 'post-norm':
            self.layers = nn.ModuleList([Block_PostNorm(idx, cfg) for idx in range(cfg.n_layers)])
        elif ln_config == "double-norm":
            self.layers = nn.ModuleList([Block_DoubleNorm(idx, cfg) for idx in range(cfg.n_layers)])
        elif ln_config == "peri-norm":
            self.layers = nn.ModuleList([Block_PeriNorm(idx, cfg) for idx in range(cfg.n_layers)])
        elif ln_config == 'mix-norm':
            self.layers = nn.ModuleList([Block_PostNorm(idx, cfg) if idx < math.floor(cfg.mixNorm_ratio * cfg.n_layers)
                                         else Block_PreNorm(idx, cfg) for idx in range(cfg.n_layers)])
            if cfg.mixNorm_ratio < 1.0:
                self.out_norm = _get_ln_variant(cfg, layer_idx=-1)
        else:
            raise ValueError("Invalid cfg.ln_config value. Choose from 'pre-norm', 'post-norm', 'double-norm', 'peri-norm', 'mix-norm'")
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

        self.freqs_cis = precompute_freqs_cis(head_dim, cfg.seq_len, 500000)[0:cfg.seq_len]

        # init all weights, scale residual branches
        self.apply(self._init_weights)
        self._scale_residual_branches()

        if cfg.tie_embeddings:
            self.tie_weights()

    def forward(self, x):
        # x: (bsz, seqlen)
        seqlen = x.size(1)
        x = self.embed_tokens(x)  # (bsz, seqlen, dim)
        freqs_cis = self.freqs_cis[:, :seqlen].to(x.device)
        for layer in self.layers:
            x = layer(x, freqs_cis) # (bsz, seqlen, dim)
        if self.out_norm is not None:
            x = self.out_norm(x)
        return self.lm_head(x) # (bsz, seqlen, vocab_size)

    def _init_weights(self, module):
        if self.weight_init.lower() in ('gpt2_res-scale', 'gpt2_no-scale', 'mix-norm'):
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif self.weight_init.lower() == 'deepnet':
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_normal_(module.weight, gain=1.0)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):  # DeepNet doesn't specify embedding initialisation, so we default to GPT-2 init
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        else:
            raise ValueError("Invalid cfg.weight_init value. Choose from 'gpt2_res-scale', 'gpt2_no-scale', 'deepnet', 'mix-norm'")

    def _scale_residual_branches(self):
        if self.weight_init.lower() == 'gpt2_res-scale':
            with torch.no_grad():
                for n, p in self.named_parameters():
                    if n.endswith('fc2.weight'): # mlp/glu output layer
                        p.mul_(1/math.sqrt(2 * self.n_layers))
                    if n.endswith('w_out.weight'): # attn output layer
                        p.mul_(1/math.sqrt(2 * self.n_layers))
        elif self.weight_init.lower() == 'gpt2_no-scale':
            return  # no residual scaling at initialisation
        elif self.weight_init.lower() == 'deepnet':
            with torch.no_grad():
                beta = (8*self.n_layers)**(-1/4)
                d = self.dim
                for n, p in self.named_parameters():
                    if n.endswith('fc1.weight') or n.endswith('fc2.weight') or n.endswith("w_out.weight"):
                        p.mul_(beta)
                    if n.endswith("w_qkv.weight"):
                        p[2*d:3*d, :].mul_(beta)  # rescale only the value projection
        elif self.weight_init.lower() == 'mix-norm':
            with torch.no_grad():
                n_pre_norm_layers = sum(isinstance(layer, Block_PreNorm) for layer in self.layers)
                if n_pre_norm_layers == 0:
                    return
                scale = 1 / math.sqrt(2 * n_pre_norm_layers)
                for layer in self.layers:
                    if isinstance(layer, Block_PreNorm):
                        layer.mlp.fc2.weight.mul_(scale)  # mlp/glu output layer
                        layer.attn.w_out.weight.mul_(scale)  # attn output layer
        else:
            raise ValueError("Invalid cfg.weight_init value. Choose from 'gpt2_res-scale', 'gpt2_no-scale', 'mix-norm'")

    def tie_weights(self):
        self.lm_head.weight = self.embed_tokens.weight

    def count_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.embed_tokens.weight.numel()
            if not self.lm_head.weight is self.embed_tokens.weight:  # if no weight tying
                n_params -= self.lm_head.weight.numel()
        return n_params
