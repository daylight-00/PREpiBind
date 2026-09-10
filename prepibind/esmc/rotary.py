# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# NOTE: this implementation is from LLaMA 2:
# https://huggingface.co/togethercomputer/LLaMA-2-7B-32K/blob/08639a72e17836184096ae6a7e2766f2a34c3e36/modeling_flash_llama.py#L114
# Flash attention rotary implementation can be installed like so: `pip install git+https://github.com/HazyResearch/flash-attention.git#subdirectory=csrc/rotary`

# Vendored from esm 3.4.0, esm/layers/rotary.py. MIT, Copyright 2026 Chan Zuckerberg Biohub,
# Inc. -- see LICENSE-esm.md. The EleutherAI/HuggingFace Apache-2.0 header above is a separate
# grant that came with the file upstream; it is reproduced verbatim and is unaffected.
# Stripped: the `interleaved`, `scale_base` (XPos) and `pos_idx_in_fp32` options, none of which
# ESMC ever sets. einops is replaced by plain torch (see apply_rotary_emb_torch). 3.4.0 changed
# nothing here beyond import formatting and type-checker pragmas.

import torch


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb_torch(x, cos, sin):
    """
    x: (batch_size, seqlen, nheads, headdim)
    cos, sin: (seqlen, rotary_dim / 2)
    """
    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= x.shape[-1]
    seqlen = x.size(1)
    cos = cos[:seqlen]
    sin = sin[:seqlen]
    # upstream: einops.repeat(cos, "s d -> s 1 (2 d)"). The new axis of size 2 is the OUTER one,
    # so this is a concatenation, not an interleave -- repeat_interleave here gives wrong numbers.
    cos = torch.cat([cos, cos], dim=-1).unsqueeze(1)
    sin = torch.cat([sin, sin], dim=-1).unsqueeze(1)
    return torch.cat(
        [x[..., :ro_dim] * cos + rotate_half(x[..., :ro_dim]) * sin, x[..., ro_dim:]],
        dim=-1,
    )


class RotaryEmbedding(torch.nn.Module):
    """
    The rotary position embeddings from RoFormer_ (Su et. al).
    A crucial insight from the method is that the query and keys are
    transformed by rotation matrices which depend on the relative positions.

    .. _RoFormer: https://arxiv.org/abs/2104.09864
    """

    def __init__(self, dim: int, base=10000.0, scaling_factor=1.0, device=None):
        super().__init__()
        self.dim = dim
        self.base = float(base)
        self.scaling_factor = scaling_factor
        self.device = device

        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        self.reset_parameters()

    def reset_parameters(self):
        inv_freq = self._compute_inv_freq(self.device)
        # persistent=False: contributes no state_dict key, which is what keeps strict=True loading
        # of the released 308-key checkpoint working.
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # Upstream's XPos scale, always None for ESMC. A None buffer is not saved either.
        self.register_buffer("scale", None)

    def _compute_inv_freq(self, device=None):
        return 1 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, device=device, dtype=torch.float32)
                / self.dim
            )
        )

    def _update_cos_sin_cache(self, seqlen, device=None, dtype=None):
        # Reset the tables if the sequence length has changed, if we're on a new device, or if
        # we're switching from inference mode to training.
        if (
            seqlen > self._seq_len_cached
            or self._cos_cached is None
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
            or (self.training and self._cos_cached.is_inference())
        ):
            self._seq_len_cached = seqlen
            # fp32 here, not the model dtype: t and inv_freq get large and half precision would
            # move cos/sin enough to change the embeddings. Do not "simplify" this.
            t = torch.arange(seqlen, device=device, dtype=torch.float32)
            t /= self.scaling_factor
            inv_freq = self.inv_freq
            if inv_freq.dtype != torch.float32:
                inv_freq = inv_freq.to(torch.float32)
            # Don't do einsum, it converts fp32 to fp16 under AMP
            freqs = torch.outer(t, inv_freq)
            self._cos_cached = torch.cos(freqs).to(dtype)
            self._sin_cached = torch.sin(freqs).to(dtype)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        q: (batch, seqlen, nheads, headdim)
        k: (batch, seqlen, nheads, headdim)
        """
        self._update_cos_sin_cache(q.shape[1], device=q.device, dtype=q.dtype)
        assert self._cos_cached is not None
        assert self._sin_cached is not None
        return (
            apply_rotary_emb_torch(q, self._cos_cached, self._sin_cached),
            apply_rotary_emb_torch(k, self._cos_cached, self._sin_cached),
        )


class TritonRotaryEmbedding(RotaryEmbedding):
    """Rotary for the flash-attn varlen path. Used only when use_flash_attn=True."""

    def forward(self, qkv: torch.Tensor, cu_seqlens, max_seqlen) -> torch.Tensor:
        """
        qkv: (n, 3, nheads, headdim)
        cu_seqlens: cumulative sequence lengths
        max_seqlen: max sequence length
        """
        # Imported here, not at module scope, so the package imports without flash-attn installed.
        from flash_attn.ops.triton.rotary import apply_rotary as apply_triton_rotary

        self._update_cos_sin_cache(max_seqlen, device=qkv.device, dtype=qkv.dtype)
        assert self._cos_cached is not None
        assert self._sin_cached is not None

        # In-place modification
        apply_triton_rotary(
            qkv[:, 0],
            self._cos_cached,
            self._sin_cached,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            inplace=True,
        )
        apply_triton_rotary(
            qkv[:, 1],
            self._cos_cached,
            self._sin_cached,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            inplace=True,
        )

        return qkv
