# Embedding Generation

This directory contains scripts to generate protein embeddings from multiple backends. These embeddings are used as input features for PREpiBind training.

> **Note:** Only the embedding generation code is provided here. Installation of each tool's dependencies (model weights, sequence databases, etc.) is left to the user. Refer to each tool's official documentation for setup instructions.

---

## Overview

| Backend | Type | Single | Pair | Requires Modified Repo |
|---------|------|:------:|:----:|:----------------------:|
| [ESM C 300M](esm/) | PLM | O | X | |
| [ESM3 Small](esm/) | PLM (API) | O | X | |
| [AlphaFold 3](af3/) | Structure | O | O | O |
| [Boltz](boltz/) | Structure | O | O | O |
| [Chai-Lab](chai/) | Structure | O | O | O |

- **Single**: Per-residue embedding of shape `(L, D)`
- **Pair**: Pairwise embedding of shape `(L, L, D)`

---

## Pair to Side-chain Conversion

Structure prediction models (AF3, Boltz, Chai) output both single and pair representations. Since pair embeddings `(L, L, D)` are too large to use directly, they are converted to side-chain representations using [pair2side.ipynb](pair2side.ipynb):

```
pair (L, L, D) → mean over axis 0 (L, D) + mean over axis 1 (L, D) → concatenate → side (L, 2D)
```

Run `pair2side.ipynb` after generating pair embeddings to produce the `*_pair_side_*.h5` files expected by the training configs.

---

## Backends

### ESM (`esm/`)

Generates per-residue embeddings using ESM protein language models. Two variants are provided:

- **`esm_local_esmc_300m.py`** — Runs ESM C 300M locally on GPU. Single-threaded.
- **`esm_api_esm3_small_2408.py`** — Calls ESM3 Small via the [Forge API](https://forge.evolutionaryscale.ai). Multiprocessing with automatic retry.

```bash
cd esm
pip install esm
python esm_local_esmc_300m.py   # Local ESM C 300M
python esm_api_esm3_small_2408.py  # ESM3 API (requires API token)
```

Both scripts read sequences from `../../data/mhc_mapping/HLA2_IMGT_light.csv` by default. Toggle the commented lines in each script to switch between HLA and epitope embedding generation.

**Output:** `emb_hla_esmc_300m.h5`, `emb_hla_esm3_small.h5` (or `emb_epi_*` for epitopes)

---

### AlphaFold 3 (`af3/`)

> **Requires cloning the modified repository:** `https://github.com/daylight-00/alphafold3`

The modified repo exposes intermediate single and pair representations via a custom inference script (`run_alphafold_custom.py`).

The pipeline has three stages:

1. **MSA preparation** — Run the data pipeline with jackhmmer to build MSAs for HLA sequences
2. **HLA inference** — Run inference with prepared MSAs to extract embeddings
3. **Epitope inference** — Run inference without MSA for epitope sequences

```bash
cd af3
bash af3.sh
```

**Output:** Single and pair embeddings saved by the modified inference script. Run `pair2side.ipynb` on pair outputs.

---

### Boltz (`boltz/`)

> **Requires cloning the modified repository:** `https://github.com/daylight-00/boltz`

The modified repo exposes intermediate representations via a custom predict entrypoint (`main_test_click.py`).

```bash
cd boltz
bash boltz.sh
```

- HLA sequences use `--use_msa_server` for MSA generation.
- Epitope sequences run without MSA.

**Input:** Prepare input directories `input_boltz_hla/` and `input_boltz_epi/` following Boltz input format.
**Output:** `output_boltz_hla/`, `output_boltz_epi/`. Run `pair2side.ipynb` on pair outputs.

---

### Chai-Lab (`chai/`)

> **Requires cloning the modified repository:** `https://github.com/daylight-00/chai-lab`

The modified repo exposes trunk single and pair representations via `chai_lab.chai1_custom.run_inference`.

The pipeline has three stages:

1. **MSA preparation** (`chai_make_a3m.py`) — Runs jackhmmer against four sequence databases (uniref90, uniprot, bfd, mgnify) and reformats alignments to A3M. Requires [HMMER](http://hmmer.org/) and the corresponding databases.
2. **HLA embedding** (`chai_hla.py`) — Generates HLA embeddings using prepared MSAs.
3. **Epitope embedding** (`chai_epi.py`) — Generates epitope embeddings without MSA.

```bash
cd chai
bash chai.sh
```

**Output:** `emb_hla_chai_single.h5`, `emb_hla_chai_pair.h5`, `emb_epi_chai_single.h5`, `emb_epi_chai_pair.h5`. Run `pair2side.ipynb` on pair outputs.

---

## Output Format

All embeddings are stored in HDF5 (`.h5`) files with the following structure:

```
file.h5
├── "HLA-DRA*01:01"  → numpy array (L, D)   # single
├── "HLA-DRB1*01:01" → numpy array (L, D)
└── ...
```

For pair embeddings, arrays have shape `(L, L, D)`. After `pair2side` conversion, the shape becomes `(L, 2D)`.
