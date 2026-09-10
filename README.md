# PREpiBind: Protein Representation-integrated Epitope-MHC Class II Binding Prediction

![banner](banner.png)

> **PREpiBind: Protein Representation-integrated Epitope-MHC Class II Binding Prediction**
> David Hyunyoo Jang, Dongwoo Kim, Byungho Park, Untaek Hwang, Yoonjoo Choi, Juyong Lee.
> *bioRxiv* (2026) — Paper (coming soon) · [HuggingFace Models](https://huggingface.co/daylight-00/prepibind)

PREpiBind predicts MHC class II–peptide binding from pre-trained protein language model (PLM)
representations. It encodes epitope sequences on the fly with
[ESMC 300M](https://huggingface.co/daylight-00/esmc-300m-2024-12) and reads pre-computed HLA
embeddings for the alpha and beta chains, feeding both into a lightweight cross-attention head.

This repository is both the released model and the artifact for the paper: every dataset, figure
and table can be rebuilt from what is here. See [Reproducing the paper](#reproducing-the-paper).

---

## Quick Start

### Option A — Local Jupyter Notebook

```bash
git clone https://github.com/daylight-00/PREpiBind
cd PREpiBind
pip install --extra-index-url https://download.pytorch.org/whl/cu126 -e ".[demo]"
```

One environment, the same one the paper uses. `[demo]` adds only `huggingface_hub`, to download the
weights. The extra index is where the pinned `torch==2.6.0+cu126` wheel lives; PyPI does not carry
it, so drop the flag only if you already have a matching torch.

Open `demo/run.ipynb` and run all cells. Cell 1 downloads the weights from HuggingFace, cell 2 runs
inference and writes to `demo/outputs/`.

### Option B — Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/daylight-00/PREpiBind/blob/main/demo/run_colab.ipynb)

An interactive widget for entering allele pairs and epitope sequences. No local setup and no
install: the notebook sparse-clones only what the demo needs — `prepibind/`, `configs/predict/` and
all of `demo/` — and uses the `torch` and `huggingface_hub` that Colab already has.

### Option C — CLI

```bash
python -m prepibind.inference configs/predict/config_demo.py --plot
```

Predictions land in `outputs/prediction.csv`. Pass a different config to select a model. Every
option below overrides the same key in the config; `--help` prints the same list.

```text
config_path       the config.py to run (positional)

--batch_size      batch size
--chkp_path       PREpiBind model checkpoint
--out_path        directory for prediction.csv and plot.png
--hla_path        HLA mapping CSV
--test_path       input data CSV
--num_workers     DataLoader workers
--use_compile     use torch.compile
--plot            save the KDE plot of prediction scores
--hla_emb_path    HLA embedding HDF5 store
--esm_chkp_path   ESMC backbone checkpoint
```

Weights are fetched by the notebooks; to get them directly:

```python
from huggingface_hub import hf_hub_download

# ESMC backbone, always required
hf_hub_download(repo_id="daylight-00/esmc-300m-2024-12", filename="esmc_300m_2024_12_v0_fp16.pth", local_dir="models")

# PREpiBind checkpoint — download the one(s) you need. These are the float16 files the predict
# configs load.
hf_hub_download(repo_id="daylight-00/prepibind-demo", filename="prepibind_qualitative_s100_f0_fp16.pt", local_dir="models")  # qualitative (default)
hf_hub_download(repo_id="daylight-00/prepibind-demo", filename="prepibind_ms_s128_f3_fp16.pt",          local_dir="models")  # mass spectrometry
hf_hub_download(repo_id="daylight-00/prepibind-demo", filename="prepibind_ic50_500_s128_f2_fp16.pt",    local_dir="models")  # IC50 < 500 nM
hf_hub_download(repo_id="daylight-00/prepibind-demo", filename="prepibind_ic50_1000_s42_f1_fp16.pt",    local_dir="models")  # IC50 < 1000 nM
```

Three repositories, one job each:

| Repository | What is in it |
|---|---|
| [`daylight-00/prepibind`](https://huggingface.co/daylight-00/prepibind) | the research checkpoints: float32, optimizer state dropped. Same four names without the `_fp16` suffix. |
| [`daylight-00/prepibind-demo`](https://huggingface.co/daylight-00/prepibind-demo) | the same four checkpoints in float16, what the demo and the predict configs use |
| [`daylight-00/esmc-300m-2024-12`](https://huggingface.co/daylight-00/esmc-300m-2024-12) | the ESM C 300M backbone weights |

The full-length float32 HLA embedding store is a dataset, not a model:
[`daylight-00/prepibind-embeddings`](https://huggingface.co/datasets/daylight-00/prepibind-embeddings).

---

## Available Models

Four models, one per measurement type.

| Config                               | Trained on                           | Checkpoint (float16)                  | val_loss |
|--------------------------------------|--------------------------------------|---------------------------------------|---------:|
| `configs/predict/config_demo.py`     | qualitative binding assays (default) | `prepibind_qualitative_s100_f0_fp16.pt` |  0.35582 |
| `configs/predict/config_ms.py`       | mass-spectrometry eluted ligands     | `prepibind_ms_s128_f3_fp16.pt`          |  0.12274 |
| `configs/predict/config_ic50_500.py` | IC50, binder threshold < 500 nM      | `prepibind_ic50_500_s128_f2_fp16.pt`    |  0.46631 |
| `configs/predict/config_ic50_1000.py`| IC50, binder threshold < 1000 nM     | `prepibind_ic50_1000_s42_f1_fp16.pt`    |  0.49456 |

**How these four were chosen.** Each arm was trained as 15 runs: three seeds (42, 100, 128) by five
cross-validation folds. Every number reported in the paper is a mean over those 15 runs. A released
checkpoint is one single run — the one with the **lowest validation loss** in its arm, the value in
the table above — and its seed and fold are in the filename as `_s<seed>_f<fold>`. So a released
checkpoint will not reproduce a paper number exactly: the paper reports the mean of the fifteen, and
this is the best one of them by validation loss. Validation loss is the only selection criterion
used; no test-set quantity took part.

> Both notebooks default to the qualitative model. To use another, set `model_arm` to `ms`,
> `ic50_500` or `ic50_1000`: it is the first setting in cell 1 of `demo/run.ipynb`, under the
> "Model selection" banner, and a dropdown in the "1. Setup" cell of `demo/run_colab.ipynb`. One
> knob picks both the predict config and the checkpoint to download.

---

## Input Format

Two columns: an MHC allele pair and an epitope sequence. The pair is two allele names joined by an
underscore, **beta chain first**, and both names must match keys in the HLA mapping file the config
points at (`demo/data/mhc_mapping_demo.csv` for the demo configs, `hla_path` in any other config).

```csv
MHC,Epitope
HLA-DRB1*01:01_HLA-DRA*01:01,AAAAAYEAAFAATVP
HLA-DRB5*01:01_HLA-DRA*01:01,AAAAGWQTLSAALDA
```

`demo/data/dataset_demo.csv` is a full example: 48,352 rows, all of them `beta_alpha`. It carries
`MHC_alpha` and `MHC_beta` columns too, which the configs do not read. Both chains are looked up by
name, so the two orders are not different inputs to the model — but everything shipped here is
beta-first, and both notebooks document it that way, so an input that matches is one less thing to
check.

## HLA Allele Coverage

| Set             | Alleles | File                                        | Notes                             |
|-----------------|--------:|---------------------------------------------|-----------------------------------|
| Demo (bundled)  |     116 | `demo/data/emb_hla_esmc_small_demo_fp16.h5` | 16 MB, in this repository         |
| Research        |     154 | `emb_hla_esmc_small_0430.h5`                | 143 MB, float32, full-length; on HuggingFace as [`prepibind-embeddings`](https://huggingface.co/datasets/daylight-00/prepibind-embeddings) |

The demo set is every allele the four dataset arms use. It is derived from the research store by
`demo/build_demo_assets.py`, which cuts each chain to its peptide-binding window and casts to
float16; `python demo/build_demo_assets.py hla-store --check` verifies the shipped file against
that derivation.

## Two ways to run inference

The paper's path and the demo's path differ, and the difference is in the data, not in a fork of
the code:

|                | paper                                   | demo                                  |
|----------------|-----------------------------------------|---------------------------------------|
| HLA store      | full-length, float32                    | cut to the window, float16            |
| mapping table  | carries the window as `sequence\|start\|end` | no window; the store is already cut |
| precision      | `as-trained` — bfloat16 ESMC, float32 head | `fp16` — half throughout           |
| epitopes       | read from a precomputed store            | encoded with ESMC at run time         |

`encoder.get_plm_emb` slices only when the mapping asks it to, and raises if a window is applied to
a store that was already cut. Because the demo encodes epitopes at run time it cannot reproduce the
published numbers bit for bit whatever precision it uses, so it takes the smaller footprint. Measured
on an H100 over the first 2,000 rows of `demo/data/dataset_demo.csv`, at the batch size the config
ships (512): peak GPU memory 3.13 GiB against 5.02 GiB, and scores that move by 0.0031 on average
(worst 0.063) with Spearman 0.99985 against the `as-trained` path. Peak memory scales with the batch:
1.95 against 2.95 GiB at batch size 256, 1.36 against 1.91 at 128. The score figures do not — they
are identical to five decimals at every batch size.

Neither path needs a GPU, and `resolve_runtime` says out loud whatever it had to drop. On a
pre-Ampere CUDA card such as Colab's T4 it turns flash-attn off and, for the `as-trained` path,
runs ESMC in float16 instead of bfloat16; the `fp16` path is unchanged. Only when there is no CUDA
device at all does everything fall back to float32.

## Output

| File                     | Description                                                     |
|--------------------------|-----------------------------------------------------------------|
| `outputs/prediction.csv` | the input with `Logits` and `Score` (sigmoid) appended           |
| `outputs/plot.png`       | KDE of the prediction scores, with `--plot`                      |

---

## Repository layout

```
prepibind/            the package: model, encoders, dataprovider, training and inference
configs/train/        one config per representation (11), plus config_global.json
configs/predict/      the four released models
data/                 the four dataset arms, the HLA mapping tables, the epitope key list
pipeline/preprocess/  IEDB export -> the four arms, as notebooks             (README)
pipeline/embeddings/  one directory per backend: ESMC, ESM3, Chai-1, Boltz, AF3  (README)
analysis/             scoring and figures; the raw-prediction manifest        (README)
analysis/scoring/     raw predictions -> the *_results.csv the figures read
analysis/figures/     the figures and tables in the paper
models/               downloaded checkpoints (git-ignored)
supplementary_data/   the machine-readable D01-D12 tables                     (README)
demo/                 the notebooks in Quick Start
```

## Reproducing the paper

Two tiers, differing in what you have to download.

```bash
make figures        # every figure and table, from this repository alone. No GPU, no snapshot.
make datasets       # the four arms, from the IEDB export      (needs PREPIBIND_IEDB_EXPORT)
make supplementary  # the D01-D12 tables
make verify         # checksums for the datasets and the D tables  (needs PREPIBIND_EMB_ROOT)
make scoring        # re-derive the scoring tables from the raw predictions (needs PREPIBIND_RAW_ROOT)
```

`make figures` is the tier that matters for reading the paper: the scoring outputs are tracked, so
the figure notebooks read them directly and nothing upstream has to run first. It takes about a
minute.

`make scoring` needs the 1.2 GB prediction snapshot, unpacked, with `PREPIBIND_RAW_ROOT` pointing
at it; `analysis/README.md` explains the snapshot and the provenance manifest. `make datasets` needs the 7.7 GB IEDB export; see
`pipeline/preprocess/README.md`.

`make verify`'s last step re-derives the bundled demo HLA store from the full-length research store,
so it needs `PREPIBIND_EMB_ROOT` pointing at the directory that holds
`emb_hla_esmc_small_0430.h5` — the [`prepibind-embeddings`](https://huggingface.co/datasets/daylight-00/prepibind-embeddings)
dataset above. `make demo-assets` reads the same variable, and so does every `configs/train/*.py`
that loads an embedding store — nine of the eleven; the BLOSUM62 and DeepNeo baselines encode from
a matrix and need nothing. It has no default: the HDF5 stores are in no fresh clone, so an unset
variable is an error that names it rather than a path that was guessed.

```bash
PREPIBIND_EMB_ROOT=/path/to/stores make verify
```

What is **not** reproducible from this repository alone: training runs and embedding generation,
both of which need GPUs and, for the embeddings, one environment per backend
(`pipeline/embeddings/*/pixi.lock`). The code for both is here; the inputs and outputs are not.
Model training is also not bit-reproducible across GPU models, so numbers from a retrained
checkpoint will differ in the last decimals.

## Requirements

The paper environment is locked with [pixi](https://pixi.sh):

```bash
pixi install            # default: training, analysis, figures
pixi install -e umap    # adds RAPIDS, only needed to recompute Supplementary Figure S2
```

or, with pip on Python 3.11+:

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cu126 -e .
```

The extra index is required: `torch` is pinned to `2.6.0+cu126`, the build the released checkpoints
and the reported scores came out of, and that build is not on PyPI. pixi reads the same index from
`pyproject.toml`; pip does not.

One environment, not two. The demo used to need a separate Python 3.11 environment for `esm`, which
pins `numpy<2` and ships no 3.13 wheel. The ESM C encoder is now vendored into `prepibind/esmc/`, so
the demo runs wherever the rest of the package runs: `torch` plus `huggingface_hub`, which is all
the `[demo]` extra installs and both of which Colab already has.

|          | Minimum | Recommended |
|----------|---------|-------------|
| Python   | 3.11+   | 3.13        |
| GPU VRAM | 4 GB    | 8 GB+       |
| RAM      | 8 GB    | 16 GB+      |

GPU VRAM is for the demo's `fp16` path at a batch size of 128-256. The shipped
`batch_size: 512` peaks at 3.13 GiB in `fp16` and 5.02 GiB `as-trained`, measured on an H100 over
2,000 rows; lower `--batch_size` if that does not fit. CPU-only inference works but is slow on
large inputs.

---

## Citation

The preprint is not posted yet; the DOI will be added here when it is.

```bibtex
@article{jang2026prepibind,
  title   = {PREpiBind: Protein Representation-integrated Epitope-MHC Class II Binding Prediction},
  author  = {Jang, David Hyunyoo and Kim, Dongwoo and Park, Byungho and Hwang, Untaek and Choi, Yoonjoo and Lee, Juyong},
  journal = {bioRxiv},
  year    = {2026}
}
```

## License

MIT, see [LICENSE](LICENSE). `prepibind/esmc/` is vendored from `esm` 3.4.0 and is MIT as well,
but under a different copyright holder — "Copyright 2026 Chan Zuckerberg Biohub, Inc." — so its
licence travels with it in `prepibind/esmc/LICENSE-esm.md`. Third-party models, weights and tools
this repository builds on carry their own terms; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
