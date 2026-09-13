# Embedding generation

The main environment at the repository root trains PREpiBind, evaluates it and produces every
figure. It reads embeddings from HDF5 stores and never runs a protein language model or a structure
predictor. Producing those stores is a separate job, and each producer needs its own interpreter.

| backend | environment | Python | single `(L, D)` | pair `(L, L, D)` | why it cannot share the main environment |
|---|---|---|:---:|:---:|---|
| ESMC 300M, ESM3 Small | `esm/` | 3.13 | ✓ | | NumPy 1.26 vs the main environment's 2.2; needs `flash-attn` |
| AlphaFold 3 | `af3/` | 3.11 | ✓ | ✓ | own interpreter; JAX 0.4.34 and CUDA 12.6 toolkit |
| Boltz-1 | `boltz/` | 3.12 | ✓ | ✓ | own interpreter |
| Chai-1 | `chai/` | 3.12 | ✓ | ✓ | own interpreter; needs `hhsuite` |

Each is a pixi workspace; conda appears only where a component is genuinely not on PyPI — the
interpreter itself, `hhsuite`, `cuda-toolkit`.

```bash
cd pipeline/embeddings/esm && pixi install && pixi run python esm_local_esmc_300m.py
```

Weights and databases are **not** included. AlphaFold 3 parameters require a separate grant from
Google DeepMind; Chai-1, Boltz-1 and ESMC weights come from their own channels; the MSA databases
(BFD, UniRef90, UniProt, MGnify) are large public downloads.

## What is reproducible, and what is not

Only the ESMC path is byte-reproducible, and only on an H100-class card with the pinned stack. Each
element of the recipe matters:

```python
with torch.device(device):
    model = ESMC(960, 15, 30, tokenizer, use_flash_attn=True).eval()
model.load_state_dict(torch.load(weights, map_location=device))
model = model.to(torch.bfloat16)          # esm's from_pretrained does this; it is not optional
emb = out.embeddings.half().cpu().numpy()[1:-1, :][start_idx:end_idx, :]
```

| change | effect on the embedding |
|---|---|
| RTX 4070 Ti instead of H100 (torch 2.7.1) | 3.7 % mean relative |
| `esm` 3.4.0, everything else held fixed | 3.3 % mean relative |
| float32 instead of bfloat16 | differs |
| model built on CPU then moved to the device | differs |
| same recipe, run twice on H100 | **byte-identical** |

For the structure predictors we pin the fork commit, which fixes the algorithm, and make no
numerical claim. Chai-1 is **not** reproducible on identical input: two runs of the same sequence
with the same MSA differ by about 3 % relative. ESM3 Large differs by about 0.4 %. AlphaFold 3 and
Boltz-1 were byte-identical across the runs we measured, though not across driver or card
generations.

## Backends

The three structure predictors run from forks pinned to a full commit SHA
(`daylight-00/{alphafold3,boltz,chai-lab}`). The forks exist only to expose intermediate
representations and bypass the diffusion modules.

| backend | entry point | stages |
|---|---|---|
| ESM | `esm/esm_local_esmc_300m.py`, `esm/esm_api_esm3_small_2408.py` | local ESMC on GPU; ESM3 through the hosted API, multiprocessing with retry |
| AlphaFold 3 | `af3/af3.sh` | MSA build with jackhmmer → HLA inference with MSAs → epitope inference without |
| Boltz | `boltz/boltz.sh` | HLA with `--use_msa_server`, epitopes without; inputs in `input_boltz_{hla,epi}/` |
| Chai-1 | `chai/chai.sh` | `chai_make_a3m.py` (jackhmmer against four databases, needs HMMER) → `chai_hla.py` → `chai_epi.py` |

The ESM scripts read sequences from `../../data/mhc_mapping/HLA2_IMGT_light.csv` by default;
toggling the commented lines switches between HLA and epitope generation.

## Output format

HDF5, one dataset per allele or epitope key:

```
file.h5
├── "HLA-DRA*01:01"  → (L, D)
├── "HLA-DRB1*01:01" → (L, D)
└── ...
```

Pair embeddings are `(L, L, D)` and too large to use directly, so `pair2side.ipynb` reduces them to
the `(L, 2D)` side-chain form the training configs expect:

```
pair (L, L, D) → mean over axis 0 (L, D) + mean over axis 1 (L, D) → concatenate → side (L, 2D)
```

Run it after generating pair embeddings to produce the `*_pair_side_*.h5` files.
