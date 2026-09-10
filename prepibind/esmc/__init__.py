"""ESMC 300M encoder, vendored so that predicting from sequence needs only torch.

Copied from `esm` 3.4.0 (Chan Zuckerberg Biohub, Inc.) and stripped to the sequence-only,
no-geometric-attention path that PREpiBind actually runs:

    esm/layers/rotary.py            -> rotary.py
    esm/layers/attention.py         -> attention.py
    esm/layers/blocks.py            -> blocks.py
    esm/layers/transformer_stack.py -> transformer_stack.py
    esm/layers/regression_head.py   -> regression_head.py
    esm/models/esmc.py              -> model.py
    esm/tokenization/sequence_tokenizer.py, esm/utils/constants/esm3.py,
    esm/utils/encoding.py           -> tokenizer.py

`esm/models/esmc.py` is 3.4.0's *legacy* ESMC. 3.4.0 also ships a new `esm/models/esmc/` package
(HuggingFace `EsmcForMaskedLM`, accelerate, Transformer Engine) whose name shadows that file on
import. The package is not vendored: the released 308-key checkpoint is laid out for the legacy
module, and the package's `ESMC` is a deprecation shim over a different model class.

3.4.0's legacy stack differs from 3.1.6's, which this package was first cut from, only by type
annotations and by an `output_attentions` flag threaded through attention, blocks, the stack and
ESMC.forward. That flag is stripped here, consistently at every level. Nothing in PREpiBind reads
attention weights, and the branch it selects recomputes attention with a hand-rolled
einsum+softmax instead of `F.scaled_dot_product_attention` -- a different kernel, so not a path
any published number came through. With it gone, each forward returns a tensor rather than a
(tensor, attn_weights) pair and ESMCOutput keeps three fields rather than 3.4.0's four; the code
that remains is the same code the 3.1.6-derived version ran.

Every module attribute name and every nn.Sequential index is unchanged, so the released 308-key
ESMC 300M checkpoint still loads with strict=True. Verified bit-identical to
`esm.models.esmc.ESMC(use_flash_attn=False)` on CPU in float32 and float16 -- embeddings,
sequence_logits and hidden_states all with max abs diff 0.0. `python -m prepibind.esmc.check_parity`
re-runs that check, and skips itself where `esm` is not installed.

MIT, but not under PREpiBind's own grant: `esm` 3.4.0 is Copyright 2026 Chan Zuckerberg Biohub,
Inc., under the MIT licence reproduced verbatim in LICENSE-esm.md. Same terms as the rest of this
repository, different copyright holder -- so that notice has to travel with this directory, which
is all MIT asks. (The 3.1.6 copy this was first cut from was under EvolutionaryScale's Cambrian
Open License Agreement; upstream relicensed, and re-deriving from 3.4.0 is what carries that
through.) rotary.py additionally carries EleutherAI/HuggingFace's Apache-2.0 header, a separate
grant that arrived with the file upstream and is preserved verbatim.
"""

from .model import ESMC, ESMCOutput
from .tokenizer import (
    EsmSequenceTokenizer,
    get_esmc_model_tokenizers,
    tokenize_sequence,
)

__all__ = [
    "ESMC",
    "ESMCOutput",
    "EsmSequenceTokenizer",
    "get_esmc_model_tokenizers",
    "tokenize_sequence",
]
