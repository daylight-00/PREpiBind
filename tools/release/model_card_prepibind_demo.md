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

All four are the val_loss argmin of their arm.

## How these four were chosen

Each arm was trained as 15 runs — three seeds by five cross-validation folds — and every number in
the paper is a mean over those 15. A released checkpoint is one of the fifteen, the one with the
lowest validation loss in its arm, the value in the table. No test-set quantity took part. So a
single checkpoint does not reproduce a paper number, and this float16 copy reproduces one less
exactly still; the [research card](https://huggingface.co/daylight-00/prepibind) has the detail.

## This does not reproduce the research path bit for bit

Two reasons, both accepted rather than fixed: the head and ESMC both run in float16, against a
float32 head and bfloat16 ESMC on the research path; and the demo encodes epitopes with ESMC at run
time instead of reading the precomputed store training and evaluation used, which makes bit-exactness
impossible at any precision. On Colab's free T4 flash-attn is off as well, and the runtime says so.

Measured, on 2,000 demo rows, this float16 path against the `as-trained` path:

| | |
|---|---|
| mean absolute difference in score | **0.0031** |
| worst case | **0.0619** |
| Spearman correlation | **0.99985** |
| peak GPU memory, H100 at the shipped batch size of 512 | 3.13 GiB, against 5.02 GiB |

Ranking is essentially preserved. If a decision turns on the third decimal, use the float32
checkpoints and the `as-trained` precision.

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
and is a verified derivation of the full-length research store `emb_hla_esmc_small_0430.h5`:
`PREPIBIND_EMB_ROOT=<its directory> python demo/build_demo_assets.py hla-store --check`.

## Usage

```bash
git clone https://github.com/daylight-00/PREpiBind && cd PREpiBind
pip install --extra-index-url https://download.pytorch.org/whl/cu126 -e ".[demo]"

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
