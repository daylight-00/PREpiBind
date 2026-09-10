---
license: mit
library_name: pytorch
tags:
  - biology
  - protein
  - immunology
  - mhc-class-ii
  - epitope
  - binding-prediction
  - esm
pipeline_tag: tabular-classification
---

# PREpiBind — demo checkpoints (float16)

The same four models as
[`daylight-00/prepibind`](https://huggingface.co/daylight-00/prepibind), cast to float16. 106.5 MiB
each instead of 213.0 MiB. This is what the notebooks and the Colab demo download, and what
`configs/predict/*.py` load by default.

Use these to try the model. Use the float32 ones for anything you intend to report.

- Code: <https://github.com/daylight-00/PREpiBind>
- Colab: `demo/run_colab.ipynb` in that repository
- Backbone: [`daylight-00/esmc-300m-2024-12`](https://huggingface.co/daylight-00/esmc-300m-2024-12)

## The four files

| file | arm | seed | fold | val_loss | predict config |
|---|---|---|---|---:|---|
| `prepibind_qualitative_s100_f0_fp16.pt` | qualitative binding assays | 100 | 0 | 0.35582 | `configs/predict/config_demo.py` |
| `prepibind_ms_s128_f3_fp16.pt` | mass-spectrometry eluted ligands | 128 | 3 | 0.12274 | `configs/predict/config_ms.py` |
| `prepibind_ic50_500_s128_f2_fp16.pt` | IC50, binder threshold < 500 nM | 128 | 2 | 0.46631 | `configs/predict/config_ic50_500.py` |
| `prepibind_ic50_1000_s42_f1_fp16.pt` | IC50, binder threshold < 1000 nM | 42 | 1 | 0.49456 | `configs/predict/config_ic50_1000.py` |

Each is 106.5 MiB, a one-key `{"model_state_dict": ...}` dict of 64 float16 tensors, 55,820,161
parameters. Nothing is quantised or pruned: this is `tensor.half()` of the float32 release, produced
by `demo/build_demo_assets.py checkpoint`, which is a pure function of the training checkpoint and
can be re-derived at any time.

All four rows are the val_loss argmin of their arm. **Not yet published:** the files exist locally
and nothing has been uploaded to HuggingFace; the sequence for doing so is
`tools/release/upload_plan.md`.

## How these four were chosen

Each arm was trained as 15 runs: three seeds (42, 100, 128) by five cross-validation folds.
**Every number reported in the paper is a mean over those 15 runs.** A released checkpoint is one
run out of the fifteen, the one with the **lowest validation loss** in its arm — the value in the
table. **Validation loss is the selection criterion; the `Val ROC-AUC` in the training logs is
invalid — `train.py` calls its own `roc_auc_score(outputs, labels)` with the arguments swapped, so
21.7 % of the logged values fall outside [0, 1] — and it was never used.** No test-set quantity took
part in the choice either. The val_loss column was re-derived on 2026-09-10 from
`analysis/val_metrics.csv`; the numbers published before that date were minima over a filtered
subset of the epoch lines and were too high.

That correction moved one selection. The **ms** row was seed 100, fold 1 (true val_loss 0.13128)
and is now seed 128, fold 3 (0.12274) — a different run of the same fifteen the paper averages for
that arm, trained from the same 77,954-row `data/dataset/ms_ql/train.csv` under a byte-identical
config. The research card sets out why the two are comparable. The other three arms kept the run
they always had; only the number printed beside them changed.

A single checkpoint therefore does not
reproduce a paper number, and this float16 copy of it reproduces one even less exactly. The full
comparison, mean over 15 runs against this run alone, is on the
[research card](https://huggingface.co/daylight-00/prepibind).

## This does not reproduce the research path bit for bit

Two separate reasons, and both of them are accepted rather than fixed:

1. **Half precision throughout.** The head runs in float16 and so does ESMC, against float32 head
   and bfloat16 ESMC on the research path.
2. **The demo encodes epitopes with ESMC at run time** instead of reading the precomputed store the
   training and the paper's evaluation used. That alone makes bit-exactness impossible whatever
   precision it runs in. On Colab's free tier the GPU is a **T4, which is pre-Ampere, so flash-attn
   is off** and the runtime says so when it drops it.

Measured, on 2,000 demo rows, this float16 path against the `as-trained` path:

| | |
|---|---|
| mean absolute difference in score | **0.0031** |
| worst case | **0.0619** |
| Spearman correlation | **0.99985** |
| peak GPU memory | 1.95 GiB, against 2.95 GiB |

Ranking is essentially preserved; individual scores move in the third decimal, and one row in two
thousand moved by 0.06. If a decision turns on that, use the float32 checkpoints and the
`as-trained` precision.

## Input format

Two columns. The MHC field is the two chain names joined by an underscore, and both must be keys in
the mapping table the config points at. The datasets and the demo write **beta first, then alpha**;
the order does not change the score, the head having no positional encoding, but it keeps inputs
comparable with the published ones.

```csv
MHC,Epitope
HLA-DRB1*01:01_HLA-DRA*01:01,PKYVKQNTLKLATA
HLA-DPB1*03:01_HLA-DPA1*01:03,AAAARYPNVTIAAAA
H2-IAbB_H2-IAbA,AAAAPAAAATTAAPA
```

Output is the input with `Logits` and `Score` (sigmoid) appended. Every training epitope is a
15-mer; other lengths run but are outside the training distribution.

The demo ships its own HLA store, `demo/data/emb_hla_esmc_small_demo_fp16.h5` (16 MB, float16, 116
alleles, cut to the peptide-binding window) - every allele the four dataset arms use, which is 98
human HLA class II chains and 18 mouse H2 chains. That file is in the GitHub repository, not here,
and is a verified derivation of the full-length research store:
`python demo/build_demo_assets.py hla-store --check`.

## Usage

```bash
git clone https://github.com/daylight-00/PREpiBind && cd PREpiBind
pip install --extra-index-url https://download.pytorch.org/whl/cu126 -e .

hf download daylight-00/prepibind-demo      prepibind_qualitative_s100_f0_fp16.pt --local-dir models
hf download daylight-00/esmc-300m-2024-12   esmc_300m_2024_12_v0_fp16.pth         --local-dir models

python -m prepibind.inference configs/predict/config_demo.py --plot
```

Predictions land in `outputs/prediction.csv`; `--plot` also writes a KDE of the scores. The config
already points at `models/prepibind_qualitative_s100_f0_fp16.pt`; switch models by switching config,
or override any single key:

```bash
python -m prepibind.inference configs/predict/config_ms.py --test_path my_input.csv --out_path out
```

The same thing from Python, which is what the notebooks do:

```python
from prepibind.inference import load_config, main

cfg = load_config("configs/predict/config_demo.py", test_path="my_input.csv", out_path="outputs")
df = main(cfg)
```

The ESMC encoder is vendored into `prepibind/esmc/`, so the demo needs only `torch` and
`huggingface_hub`, both already present on Colab. Do not `pip install esm`.

No GPU is required. Without CUDA the runtime falls back to float32 throughout and turns off
flash-attn; on a pre-Ampere card it turns off flash-attn and keeps float16. Either way it prints
what it changed, because a silently downgraded run is one whose numbers cannot be compared with
anything.

## Limitations

Everything on the [research card](https://huggingface.co/daylight-00/prepibind) applies — one run
rather than the reported mean, class II only, the disclosed H2 chain correction in the embedding
store, binding rather than immunogenicity — plus the precision deviation measured above.

## Licence

**MIT** for these weights, same as the repository. The ESMC 300M backbone is not ours but is also
MIT: it moved to [`biohub/esmc-300m-2024-12`](https://huggingface.co/biohub/esmc-300m-2024-12)
(ungated, tagged `mit` + `other`), and the vendored encoder source is copied from `esm` 3.4.0, whose
licence is plain MIT, "Copyright 2026 Chan Zuckerberg Biohub, Inc.". Checked 2026-09-10; see
`THIRD_PARTY_NOTICES.md` in the repository.

## Citation

```bibtex
@article{jang2026prepibind,
  title   = {PREpiBind: Protein Representation-integrated Epitope-MHC Class II Binding Prediction},
  author  = {Jang, David Hyunyoo and Kim, Dongwoo and Park, Byungho and Hwang, Untaek and Choi, Yoonjoo and Lee, Juyong},
  journal = {bioRxiv},
  year    = {2026}
}
```
