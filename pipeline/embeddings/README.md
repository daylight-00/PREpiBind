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

The scripts spell their input `../../data/...`, one `..` short of the repository's `data/` from
`pipeline/embeddings/<backend>/`; repoint it to `../../../data/...` before running.

Weights and databases are **not** included. AlphaFold 3 parameters require a separate grant from
Google DeepMind; Chai-1, Boltz-1 and ESMC weights come from their own channels; the MSA databases
(BFD, UniRef90, UniProt, MGnify) are large public downloads.

## What is reproducible, and what is not

Only the ESMC path is byte-reproducible, and only on an H100-class card with the pinned stack. Each
element of the recipe in `esm/esm_local_esmc_300m.py` matters:

```python
client = ESMC.from_pretrained("esmc_300m").to("cuda")   # from_pretrained casts to bfloat16
emb = logits_output.embeddings.cpu().detach().squeeze(0).numpy()[1:-1, :]
```

`[1:-1]` drops `[CLS]` and `[EOS]`; nothing else is cut. The peptide-binding window is not applied
here — it is carried in `data/mhc_mapping/HLA2_IMGT_MSA_idx_edit.csv` as `sequence|start|end` and
applied at load time.

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
| AlphaFold 3 | `af3/af3_make_json.ipynb`, `af3/af3.sh` | JSONs into `json_hla/` (42 shards) and `json_epi/` → `af3.sh <shard>` builds MSAs with jackhmmer into `json_msa/` → HLA inference from `input_af3_hla/`, epitopes from `input_af3_epi/` without MSAs |
| Boltz | `boltz/boltz_make_fasta.ipynb`, `boltz/boltz.sh` | FASTAs into `input_boltz_{hla,epi}/` → HLA with `--use_msa_server`, epitopes without |
| Chai-1 | `chai/chai.sh` | `chai_make_a3m.py` (jackhmmer against four databases, needs HMMER) → `chai_hla.py` → `chai_epi.py` |

Nothing here writes `input_af3_hla/` or `input_af3_epi/`: stage the MSA-enriched JSONs from
`json_msa/` and the epitope JSONs from `json_epi/` into them before the two inference stages.

The ESM scripts read sequences from `data/mhc_mapping/HLA2_IMGT_light.csv` by default; toggling the
commented lines switches to epitopes (`data/unique_epitope_whole.csv`). `esm_api_esm3_small_2408.py`
picks the hosted model with its `model=` string; the ESMC 600M, ESMC 6B, ESM3 Medium and ESM3 Large
stores that `configs/train/` names have no generator in this repository.

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

## Where the stores go

`PREPIBIND_EMB_ROOT` is the directory the training configs read; it has no default.
`pair2side.ipynb` reads `$PREPIBIND_EMB_ROOT/pair/*pair*.h5` and writes `$PREPIBIND_EMB_ROOT/side/`,
so stage the pair stores under `pair/` first and move the results up — the configs read
`*_pair_side_*.h5` from the root itself, not from `side/`.

The generators write their own names (`../emb_hla_esmc_300m.h5`, `emb_hla_esm3_small.h5`,
`../emb_hla_chai_{single,pair}.h5`), which are not the names the configs ask for; renaming is
manual:

| config | reads, relative to `$PREPIBIND_EMB_ROOT` |
|---|---|
| `config_esmc_small.py` | `emb_{hla,epi}_esmc_small_0430.h5` |
| `config_esm3_small.py` | `esm_large/emb_{hla,epi}_esm3_small_2408_0430.h5` |
| `config_af3.py` | `emb_{hla,epi}_af3_single_0430.h5`, `emb_{hla,epi}_af3_pair_side_0430.h5` |
| `config_boltz.py` | `emb_{hla,epi}_boltz_single_0430.h5`, `emb_{hla,epi}_boltz_pair_side_0430.h5` |
| `config_chai.py` | `emb_hla_chai_jack_{single,pair_side}_0430.h5`, `emb_epi_chai_esm_{single,pair_side}_0430.h5` |
