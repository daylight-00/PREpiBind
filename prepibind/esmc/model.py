# Vendored from esm 3.4.0, esm/models/esmc.py. MIT, Copyright 2026 Chan Zuckerberg Biohub, Inc.
# -- see LICENSE-esm.md.
#
# In 3.4.0 that path is the *legacy* ESMC and is shadowed at import time: `esm.models.esmc`
# resolves to the esm/models/esmc/ package (a HuggingFace EsmcForMaskedLM plus a deprecation shim
# re-exporting the names ESMC and ESMCOutput), because a package wins over a same-named module.
# The legacy file is what this package descends from and what the released 308-key checkpoint
# lays out; the package is deliberately not vendored. Read the legacy file from the sdist, or by
# path, not by importing it.
#
# Stripped: the ESMCInferenceClient base and the SDK methods built on it (encode/decode/logits),
# from_pretrained, and _tokenize/_detokenize. PREpiBind constructs ESMC directly, loads a local
# state_dict and reads .embeddings; nothing else was reachable. attr.dataclass -> dataclasses.
# `output_attentions` is stripped through the whole stack, so ESMCOutput keeps three fields;
# 3.4.0's fourth, `attentions`, could only ever be None here.
#
# from_pretrained is also where upstream's `model.to(torch.bfloat16)` lived. That cast is the
# reason the paper's stored embeddings are bfloat16; it must not come back here -- the caller
# (prepibind/inference.py) sets the dtype explicitly.

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .regression_head import RegressionHead
from .tokenizer import EsmSequenceTokenizer
from .transformer_stack import TransformerStack


def _is_flash_attn_available() -> bool:
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        return False
    return True


@dataclass
class ESMCOutput:
    sequence_logits: torch.Tensor
    embeddings: torch.Tensor | None
    hidden_states: torch.Tensor | None


class ESMC(nn.Module):
    """
    ESMC model implementation.

    Args:
        d_model (int): The dimensionality of the input and output feature vectors.
        n_heads (int): The number of attention heads in the transformer layers.
        n_layers (int): The number of transformer layers.
        use_flash_attn (bool): Use the varlen flash-attn path. Needs flash-attn installed and an
            Ampere-or-newer GPU; as upstream, it falls back to the plain path if flash-attn is
            missing. Read back `._use_flash_attn` to see which path was actually built.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        tokenizer: EsmSequenceTokenizer,
        use_flash_attn: bool = True,
    ):
        super().__init__()
        self.embed = nn.Embedding(64, d_model)

        self._use_flash_attn = bool(use_flash_attn) and _is_flash_attn_available()
        self.transformer = TransformerStack(
            d_model,
            n_heads,
            n_layers,
            use_flash_attn=self._use_flash_attn,
        )

        self.sequence_head = RegressionHead(d_model, 64)
        self.tokenizer = tokenizer

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def raw_model(self):
        return self

    def forward(
        self,
        sequence_tokens: torch.Tensor | None = None,
        sequence_id: torch.Tensor | None = None,
    ) -> ESMCOutput:
        """
        Performs forward pass through the ESMC model.

        Args:
            sequence_tokens (torch.Tensor, optional): The amino acid tokens.
            sequence_id (torch.Tensor, optional): The sequence ID.

        Returns:
            ESMCOutput: The output of the ESMC model.
        """
        if sequence_id is None:
            # For ESMC, a boolean mask is created in place of sequence_id if not specified.
            sequence_id = sequence_tokens != self.tokenizer.pad_token_id

        x = self.embed(sequence_tokens)

        B, L = x.shape[:2]

        # If sequence_id looks like a mask.
        if self._use_flash_attn:
            from flash_attn.bert_padding import unpad_input

            assert (
                sequence_id.dtype == torch.bool
            ), "sequence_id must be a boolean mask if Flash Attention is used"
            assert sequence_id.shape == (B, L)
            x, indices, *_ = unpad_input(x, sequence_id)
        else:
            indices = None

        x, _, hidden_states = self.transformer(x, sequence_id=sequence_id)

        if self._use_flash_attn:
            from flash_attn.bert_padding import pad_input

            assert indices is not None
            x = pad_input(x, indices, B, L)  # Back to [B, L, D]
            hidden_states = [
                # Back to [[B, L, D], ...]
                pad_input(h, indices, B, L)
                for h in hidden_states
            ]

        # Stack hidden states into a [n_layers, B, L, D] matrix.
        hidden_states = torch.stack(hidden_states, dim=0)  # type: ignore

        sequence_logits = self.sequence_head(x)
        output = ESMCOutput(
            sequence_logits=sequence_logits, embeddings=x, hidden_states=hidden_states
        )
        return output
