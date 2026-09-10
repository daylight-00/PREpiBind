"""Check the vendored ESMC against the `esm` package installed in this environment.

    python -m prepibind.esmc.check_parity [esmc_checkpoint.pth]

The vendored code is re-derived from `esm` 3.4.0 (MIT). The control is whatever `esm` is
installed, and 3.1.6 is a legitimate one: 3.4.0's legacy stack differs from 3.1.6's only by type
annotations and by the `output_attentions` plumbing this package strips, so matching 3.1.6 exactly
is precisely the claim. Skips itself where `esm` is not installed, because the released
environment does not have it. Compares state_dict key sets, tokenizer output, and a forward pass
in float32 and float16 -- all of which were bit-identical when the code was vendored.

From esm 3.2 on, `esm.models.esmc` resolves to a package that shadows the legacy
`esm/models/esmc.py` and exports a deprecation shim over a HuggingFace model class.
`reference_esmc()` steps around that and loads the legacy module by path, so the comparison stays
against the code this package was actually derived from.
"""

import importlib.util
import pathlib
import random
import sys

import torch

DEFAULT_CHECKPOINT = "models/esmc_300m_2024_12_v0_fp16.pth"

# 29 real symbols, the short mask, unknown characters, and the special-token strings themselves.
FUZZ_ALPHABET = (
    list("LAGVSERTIDPKQNFYMHWCXBUZO.-|")
    + ["_"]
    + list("acdefgik")
    + list("0123456789")
    + [" ", "\t", "*", "J", "j"]
    + ["<cls>", "<pad>", "<eos>", "<unk>", "<mask>"]
)


def check_tokenizer(n_cases: int = 3000) -> None:
    from esm.tokenization import EsmSequenceTokenizer as RefTokenizer
    from esm.utils import encoding as ref_encoding

    from prepibind.esmc import EsmSequenceTokenizer, tokenize_sequence

    ref, own = RefTokenizer(), EsmSequenceTokenizer()
    for name in ("pad_token_id", "mask_token_id", "cls_token_id", "eos_token_id",
                 "unk_token_id", "chain_break_token_id", "vocab_size"):
        assert getattr(ref, name) == getattr(own, name), name

    rng = random.Random(0)
    cases = [""] + [
        "".join(rng.choice(FUZZ_ALPHABET) for _ in range(rng.randint(0, 12)))
        for _ in range(n_cases - 1)
    ]
    for text in cases:
        for add_special in (True, False):
            a = ref_encoding.tokenize_sequence(text, ref, add_special_tokens=add_special)
            b = tokenize_sequence(text, own, add_special_tokens=add_special)
            assert torch.equal(a, b), (text, add_special, a.tolist(), b.tolist())
    print(f"tokenizer: {2 * len(cases)} cases, 0 mismatches")


def reference_esmc():
    """The legacy `esm` ESMC class, whichever esm version is installed."""
    import esm.models.esmc as mod

    if not (mod.__file__ or "").endswith("__init__.py"):
        return mod.ESMC  # a plain module: esm <= 3.1.x

    # esm >= 3.2: esm/models/esmc/ is a package and shadows esm/models/esmc.py, which is still
    # shipped and is the module this package descends from. Load it by path.
    legacy = pathlib.Path(mod.__file__).parent.parent / "esmc.py"
    if not legacy.is_file():
        raise RuntimeError(f"legacy ESMC module not found next to {mod.__file__}")
    spec = importlib.util.spec_from_file_location("_esm_legacy_esmc", legacy)
    assert spec is not None and spec.loader is not None
    legacy_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy_mod)
    print(f"reference: legacy ESMC loaded from {legacy}")
    return legacy_mod.ESMC


def check_forward(checkpoint: str) -> None:
    from esm.tokenization import get_esmc_model_tokenizers as ref_tokenizers

    RefESMC = reference_esmc()

    from prepibind.esmc import ESMC, get_esmc_model_tokenizers

    state = torch.load(checkpoint, map_location="cpu")
    ref = RefESMC(960, 15, 30, ref_tokenizers(), use_flash_attn=False)
    own = ESMC(960, 15, 30, get_esmc_model_tokenizers(), use_flash_attn=False)
    assert set(ref.state_dict()) == set(own.state_dict()) == set(state), "state_dict key sets differ"
    print(f"state_dict: {len(state)} keys, identical")
    ref.load_state_dict(state, strict=True), own.load_state_dict(state, strict=True)
    ref.eval(), own.eval()

    # Two epitopes of different length in one batch, so the pad mask is exercised.
    tok = get_esmc_model_tokenizers()
    seqs = ["PKYVKQNTLKLAT", "AAYSDQATPLLLS"[:9]]
    ids = [tok.encode(s, add_special_tokens=True) for s in seqs]
    width = max(len(i) for i in ids) + 4  # extra pad on both rows
    tokens = torch.full((len(ids), width), tok.pad_token_id, dtype=torch.long)
    for i, row in enumerate(ids):
        tokens[i, : len(row)] = torch.tensor(row)

    for dtype in (torch.float32, torch.float16):
        r, o = ref.to(dtype), own.to(dtype)
        with torch.no_grad():
            a, b = r(tokens), o(tokens)
        for field in ("embeddings", "sequence_logits", "hidden_states"):
            x, y = getattr(a, field), getattr(b, field)
            diff = (x.float() - y.float()).abs().max().item()
            assert torch.equal(x, y), f"{dtype} {field}: max abs diff {diff}"
        print(f"{str(dtype):>14}: embeddings, sequence_logits, hidden_states all bit-identical")


def main() -> int:
    if importlib.util.find_spec("esm") is None:
        print("esm is not installed -- nothing to compare against, skipping")
        return 0
    from importlib.metadata import PackageNotFoundError, version

    try:
        print(f"control: esm {version('esm')}")
    except PackageNotFoundError:
        print("control: esm (version unknown)")
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CHECKPOINT
    check_tokenizer()
    check_forward(checkpoint)
    print("vendored ESMC matches esm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
