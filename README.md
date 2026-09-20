# PREpiBind: Protein Representation-integrated Epitope-MHC Class II Binding Prediction

![banner](banner.png)

> **PREpiBind: Protein Representation-integrated Epitope-MHC Class II Binding Prediction**
> David Hyunyoo Jang, Dongwoo Kim, Byungho Park, Untaek Hwang, Yoonjoo Choi, Juyong Lee.
> *bioRxiv* (2026) — Paper (coming soon) · [HuggingFace Models](https://huggingface.co/daylight-00/prepibind)

PREpiBind predicts MHC class II–peptide binding from pre-trained protein language model
representations. It encodes epitopes on the fly with
[ESMC 300M](https://huggingface.co/daylight-00/esmc-300m-2024-12) and reads pre-computed HLA
embeddings for the alpha and beta chains, feeding both into a lightweight cross-attention head.

This repository is both the released model and the paper's artifact: every dataset, table and
quantitative figure can be rebuilt from what is here. See [Reproducing the paper](#reproducing-the-paper).

---

## Quick Start

### Option A — Local Jupyter Notebook

```bash
git clone https://github.com/daylight-00/PREpiBind
cd PREpiBind
pip install --extra-index-url https://download.pytorch.org/whl/cu126 -e ".[demo]"
```

`[demo]` adds only `huggingface_hub`, to download the weights. The extra index is where the pinned
`torch==2.6.0+cu126` wheel lives; drop the flag only if you already have a matching torch.

Open `demo/run.ipynb` and run all cells. Cell 1 downloads the weights, cell 2 runs inference and
writes to `demo/outputs/`.

### Option B — Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/daylight-00/PREpiBind/blob/main/demo/run_colab.ipynb)

An interactive widget for entering allele pairs and epitope sequences. No install: the notebook
sparse-clones `prepibind/`, `configs/predict/` and `demo/`, and uses the `torch` and
`huggingface_hub` Colab already has.

### Option C — CLI

```bash
python -m prepibind.inference configs/predict/config_demo.py --plot
```

Predictions land in `outputs/prediction.csv`, alongside `plot.png` with `--plot`. Pass a different
config to select a model; `--help` lists the ten config keys that have command-line flags.

Weights are fetched by the notebooks; to get them directly:

```python
from huggingface_hub import hf_hub_download

# ESMC backbone, always required
hf_hub_download(repo_id="daylight-00/esmc-300m-2024-12", filename="esmc_300m_2024_12_v0_fp16.pth", local_dir="models")

# PREpiBind checkpoint — the float16 files the predict configs load
hf_hub_download(repo_id="daylight-00/prepibind-demo", filename="prepibind_qualitative_s100_f0_fp16.pt", local_dir="models")  # qualitative (default)
hf_hub_download(repo_id="daylight-00/prepibind-demo", filename="prepibind_ms_s128_f3_fp16.pt",          local_dir="models")  # mass spectrometry
hf_hub_download(repo_id="daylight-00/prepibind-demo", filename="prepibind_ic50_500_s128_f2_fp16.pt",    local_dir="models")  # IC50 < 500 nM
hf_hub_download(repo_id="daylight-00/prepibind-demo", filename="prepibind_ic50_1000_s42_f1_fp16.pt",    local_dir="models")  # IC50 < 1000 nM
```

| Repository | What is in it |
|---|---|
| [`daylight-00/prepibind`](https://huggingface.co/daylight-00/prepibind) | the research checkpoints: float32, optimizer state dropped. Same four names without `_fp16` |
| [`daylight-00/prepibind-demo`](https://huggingface.co/daylight-00/prepibind-demo) | the same four in float16, what the demo and the predict configs use |
| [`daylight-00/esmc-300m-2024-12`](https://huggingface.co/daylight-00/esmc-300m-2024-12) | the ESMC 300M backbone weights |
| [`daylight-00/prepibind-embeddings`](https://huggingface.co/datasets/daylight-00/prepibind-embeddings) | the full-length float32 HLA store (a dataset, not a model) |

---

## Available Models

Four models, one per measurement type.

| Config                               | Trained on                           | Checkpoint (float16)                  | val_loss |
|--------------------------------------|--------------------------------------|---------------------------------------|---------:|
| `configs/predict/config_demo.py`     | qualitative binding assays (default) | `prepibind_qualitative_s100_f0_fp16.pt` |  0.35582 |
| `configs/predict/config_ms.py`       | mass-spectrometry eluted ligands     | `prepibind_ms_s128_f3_fp16.pt`          |  0.12274 |
| `configs/predict/config_ic50_500.py` | IC50, binder threshold < 500 nM      | `prepibind_ic50_500_s128_f2_fp16.pt`    |  0.46631 |
| `configs/predict/config_ic50_1000.py`| IC50, binder threshold < 1000 nM     | `prepibind_ic50_1000_s42_f1_fp16.pt`    |  0.49456 |

**How these four were chosen.** Each arm was trained as 15 runs — three seeds by five
cross-validation folds — and every number in the paper is a mean over those 15. A released
checkpoint is one run, the one with the lowest validation loss in its arm, its seed and fold in the
filename as `_s<seed>_f<fold>`. It will not reproduce a paper number exactly. Validation loss was
the only selection criterion; no test-set quantity took part.

> Both notebooks default to the qualitative model. Set `model_arm` to `ms`, `ic50_500` or
> `ic50_1000` to switch: one knob picks both the predict config and the checkpoint to download.

---

## Input Format

Two columns: an MHC allele pair and an epitope sequence. The pair is two allele names joined by an
underscore, **beta chain first**, and both must match keys in the HLA mapping file the config points
at (`demo/data/mhc_mapping_demo.csv` for the demo configs, `hla_path` otherwise).

```csv
MHC,Epitope
HLA-DRB1*01:01_HLA-DRA*01:01,AAAAAYEAAFAATVP
HLA-DRB5*01:01_HLA-DRA*01:01,AAAAGWQTLSAALDA
```

`demo/data/dataset_demo.csv` is a full example, 48,352 rows. Output is the input with `Logits` and
`Score` (sigmoid) appended.

## HLA Allele Coverage

| Set             | Alleles | File                                        | Notes                             |
|-----------------|--------:|---------------------------------------------|-----------------------------------|
| Demo (bundled)  |     116 | `demo/data/emb_hla_esmc_small_demo_fp16.h5` | 16 MB, in this repository         |
| Research        |     154 | `emb_hla_esmc_small_0430.h5`                | 143 MB, float32, full-length; on HuggingFace as [`prepibind-embeddings`](https://huggingface.co/datasets/daylight-00/prepibind-embeddings) |

The demo set is every allele the four dataset arms use, derived from the research store by
`demo/build_demo_assets.py`, which cuts each chain to its peptide-binding window and casts to
float16. `python demo/build_demo_assets.py hla-store --check` verifies the shipped file against that
derivation.

## Two ways to run inference

The paper's path and the demo's differ in the data, not in a fork of the code:

|                | paper                                        | demo                                |
|----------------|----------------------------------------------|-------------------------------------|
| HLA store      | full-length, float32                         | cut to the window, float16          |
| mapping table  | carries the window as `sequence\|start\|end` | no window; the store is already cut |
| precision      | `as-trained` — bfloat16 ESMC, float32 head   | `fp16` — half throughout            |
| epitopes       | read from a precomputed store                | encoded with ESMC at run time        |

Because the demo encodes epitopes at run time it cannot reproduce the published numbers bit for bit
at any precision, so it takes the smaller footprint: on an H100 at the shipped batch size of 512,
3.13 GiB peak against 5.02, with scores moving 0.0031 on average (Spearman 0.99985). Neither path
needs a GPU, and `resolve_runtime` says out loud whatever it had to drop — on a pre-Ampere card it
turns flash-attn off and runs ESMC in float16; with no CUDA device at all, everything falls back to
float32.

---

## Repository layout

```
prepibind/            the package: model, encoders, dataprovider, training and inference
configs/train/        one config per representation (11), plus config_global.json
configs/predict/      the four released models
data/                 the four dataset arms, the HLA mapping tables, the epitope key list
pipeline/preprocess/  IEDB export -> the four arms, as notebooks             (README)
pipeline/embeddings/  four environments: esm/ (ESMC and ESM3), chai/, boltz/, af3/  (README)
analysis/             the prediction snapshot, rawpath, raw_manifest.csv   (README)
analysis/scoring/     raw predictions -> the *_results.csv the figures read
analysis/figures/     the figures and tables in the paper
models/               downloaded checkpoints (git-ignored)
supplementary_data/   the machine-readable D01-D12 tables                     (README)
demo/                 the notebooks in Quick Start
```

## Reproducing the paper

```bash
make figures        # fig2-fig5, figS1-figS2, every table. This repository alone, no GPU, no snapshot
make mhc-alignment  # fetch the IPD-IMGT/HLA alignment (not shipped); make datasets needs it
make datasets       # the four arms, from the IEDB export       (needs PREPIBIND_IEDB_EXPORT)
make supplementary  # the D01-D12 tables
make verify         # checksums for the datasets and the D tables   (needs PREPIBIND_EMB_ROOT)
make scoring        # the scoring tables, from the raw predictions  (needs PREPIBIND_RAW_ROOT)
```

`make figures` is the tier that matters for reading the paper: the scoring outputs are tracked, so
the figure notebooks read them directly. It takes about a minute. Figure 1 is not in it: it is a
hand-drawn schematic, edited as `analysis/figures/fig1_workflow_and_architecture.fig` and shipped as
`fig1.{pdf,svg}`.

The three variables have no defaults, because the files they point at are in no fresh clone:
`PREPIBIND_RAW_ROOT` is the unpacked 1.2 GB prediction snapshot (`analysis/README.md`),
`PREPIBIND_IEDB_EXPORT` the 7.7 GB IEDB export (`pipeline/preprocess/README.md`), and
`PREPIBIND_EMB_ROOT` the directory holding `emb_hla_esmc_small_0430.h5` from
[`prepibind-embeddings`](https://huggingface.co/datasets/daylight-00/prepibind-embeddings), which
`make verify`, `make demo-assets` and nine of the eleven training configs all read.

Not reproducible from this repository alone: training runs and embedding generation, which need GPUs
and four environments (`pipeline/embeddings/*/pixi.lock`). The code is here; the inputs and outputs
are not. Training is also not bit-reproducible across GPU models.

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
came out of, and that build is not on PyPI.

|          | Minimum | Recommended |
|----------|---------|-------------|
| Python   | 3.11+   | 3.13        |
| GPU VRAM | 4 GB    | 8 GB+       |
| RAM      | 8 GB    | 16 GB+      |

VRAM is for the demo's `fp16` path at batch size 128-256; lower `--batch_size` if the shipped 512
does not fit. CPU-only inference works but is slow on large inputs.

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

MIT **for the code**, see [LICENSE](LICENSE) — `prepibind/`, `configs/`, `demo/`, `pipeline/`,
`analysis/` and `supplementary_data/`.

**The data is not MIT.** `data/`, `demo/data/` and `analysis/figures/data/` are derived from IEDB
(CC BY 4.0), IPD-IMGT/HLA and IPD-MHC (CC BY-NoDerivs) and UniProt (CC BY 4.0). The IPD-IMGT/HLA
alignment is not redistributed here — `pipeline/preprocess/fetch_mhc_alignment.py` downloads it from
[`github.com/ANHIG/IMGTHLA`](https://github.com/ANHIG/IMGTHLA) at release 3.59.0. Per-file terms are
in [data/LICENSE.md](data/LICENSE.md).

`prepibind/esmc/` is vendored from `esm` 3.4.0, MIT as well but under a different copyright holder,
so its licence travels with it in `prepibind/esmc/LICENSE-esm.md`. Third-party models, weights and
tools carry their own terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
