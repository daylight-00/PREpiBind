# Vendored from esm 3.4.0, esm/layers/attention.py. MIT, Copyright 2026 Chan Zuckerberg Biohub,
# Inc. -- see LICENSE-esm.md.
# Stripped: einops (replaced by unflatten/permute/flatten, verified bit-identical), the
# module-level flash-attn import (it now happens inside FlashMultiHeadAttention.forward), and the
# `output_attentions` branch. That branch is a hand-rolled einsum+softmax attention -- a different
# kernel from scaled_dot_product_attention, so it is not a path any published number came through
# -- and nothing here reads attention weights. Dropping it is what lets forward keep returning the
# context tensor rather than upstream's (context, attn_weights) pair; see __init__.py.

import torch
import torch.nn.functional as F
from torch import nn

from .rotary import RotaryEmbedding, TritonRotaryEmbedding


class MultiHeadAttention(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, bias: bool = False, qk_layernorm: bool = True
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads

        self.d_head = self.d_model // self.n_heads
        self.layernorm_qkv = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model * 3, bias=bias)
        )
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        if qk_layernorm:
            self.q_ln = nn.LayerNorm(d_model, bias=bias)
            self.k_ln = nn.LayerNorm(d_model, bias=bias)
        else:
            self.q_ln = nn.Identity()
            self.k_ln = nn.Identity()

        self.rotary = RotaryEmbedding(d_model // n_heads)

    def _apply_rotary(self, q: torch.Tensor, k: torch.Tensor):
        q = q.unflatten(-1, (self.n_heads, self.d_head))
        k = k.unflatten(-1, (self.n_heads, self.d_head))
        q, k = self.rotary(q, k)
        q = q.flatten(-2, -1)
        k = k.flatten(-2, -1)
        return q, k

    def forward(self, x, seq_id):
        qkv_BLD3 = self.layernorm_qkv(x)
        query_BLD, key_BLD, value_BLD = torch.chunk(qkv_BLD3, 3, dim=-1)
        # No-ops outside autocast; under autocast the LayerNorms return fp32 and these casts put
        # the model dtype back. Note key is cast to *query*'s dtype -- upstream quirk, kept.
        query_BLD, key_BLD = (
            self.q_ln(query_BLD).to(query_BLD.dtype),
            self.k_ln(key_BLD).to(query_BLD.dtype),
        )
        query_BLD, key_BLD = self._apply_rotary(query_BLD, key_BLD)

        # upstream: einops.rearrange(t, "b s (h d) -> b h s d", h=self.n_heads)
        def reshaper(t):
            return t.unflatten(-1, (self.n_heads, self.d_head)).permute(0, 2, 1, 3)

        query_BHLD, key_BHLD, value_BHLD = map(
            reshaper, (query_BLD, key_BLD, value_BLD)
        )

        if seq_id is not None:
            # Where True, enable participation in attention. seq_id is the bool pad mask, so
            # pad<->pad compares equal and pad rows attend among themselves; their outputs are
            # dropped downstream. "Fixing" that changes the published numbers.
            mask_BLL = seq_id.unsqueeze(-1) == seq_id.unsqueeze(-2)
            mask_BHLL = mask_BLL.unsqueeze(1)
        else:
            mask_BHLL = None

        if mask_BHLL is None:
            # Shortcut, if we don't use attention biases then torch
            # will autoselect flashattention as the implementation
            context_BHLD = F.scaled_dot_product_attention(
                query_BHLD, key_BHLD, value_BHLD
            )
        else:
            context_BHLD = F.scaled_dot_product_attention(
                query_BHLD, key_BHLD, value_BHLD, mask_BHLL
            )

        # upstream: einops.rearrange(context_BHLD, "b h s d -> b s (h d)")
        context_BLD = context_BHLD.permute(0, 2, 1, 3).flatten(-2, -1)

        return self.out_proj(context_BLD)


class FlashMultiHeadAttention(MultiHeadAttention):
    """Varlen flash-attn attention. Same parameters and state_dict as MultiHeadAttention; only
    reachable when ESMC is built with use_flash_attn=True, which requires flash-attn and Ampere+."""

    def __init__(
        self, d_model: int, n_heads: int, bias: bool = False, qk_layernorm: bool = True
    ):
        super().__init__(
            d_model=d_model, n_heads=n_heads, bias=bias, qk_layernorm=qk_layernorm
        )

        # Flash attention rotary.
        self.rotary = TritonRotaryEmbedding(d_model // n_heads)

    def forward(self, x, seq_id):
        # Imported here, not at module scope, so the package imports without flash-attn installed.
        # Upstream 3.4.0 does the import at module scope and catches (ImportError, RuntimeError);
        # a local import cannot leave the "installed but unusable" hole that widening was for.
        from flash_attn import flash_attn_varlen_qkvpacked_func

        assert seq_id.dtype == torch.bool

        seqlens = seq_id.sum(dim=-1, dtype=torch.int32)
        cu_seqlens = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
        max_seqlen = seqlens.max().item()

        qkv_ND3 = self.layernorm_qkv(x)

        query_ND, key_ND, value_ND = torch.chunk(qkv_ND3, 3, dim=-1)
        query_ND, key_ND = (
            self.q_ln(query_ND).to(query_ND.dtype),
            self.k_ln(key_ND).to(query_ND.dtype),
        )

        qkv_N3D = torch.stack([query_ND, key_ND, value_ND], dim=1)
        # upstream: einops.rearrange(qkv_N3D, "n a (h d) -> n a h d", h=self.n_heads)
        qkv_N3HD = qkv_N3D.unflatten(-1, (self.n_heads, self.d_head))
        qkv_N3HD = self.rotary(qkv_N3HD, cu_seqlens, max_seqlen)

        context_NHD = flash_attn_varlen_qkvpacked_func(
            qkv_N3HD, cu_seqlens, max_seqlen, softmax_scale=self.d_head**-0.5
        )
        # upstream: einops.rearrange(context_NHD, "n h d -> n (h d)")
        context_ND = context_NHD.flatten(-2, -1)

        return self.out_proj(context_ND)
