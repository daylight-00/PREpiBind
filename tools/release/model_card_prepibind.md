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

# PREpiBind — research checkpoints (float32)

Four MHC class II peptide-binding models, one per measurement type, in the precision they were
trained in. Weights only: the optimizer state and the epoch counter are dropped, nothing else is
changed.

- Code and everything needed to run these: <https://github.com/daylight-00/PREpiBind>
- float16 copies of the same four models, for the notebooks and Colab:
  [`daylight-00/prepibind-demo`](https://huggingface.co/daylight-00/prepibind-demo)
- The backbone they call: [`daylight-00/esmc-300m-2024-12`](https://huggingface.co/daylight-00/esmc-300m-2024-12)
- The full-length HLA embedding store they read:
  [`daylight-00/prepibind-embeddings`](https://huggingface.co/datasets/daylight-00/prepibind-embeddings)

## The four files

| file | arm | seed | fold | val_loss | predict config |
|---|---|---|---|---:|---|
| `prepibind_qualitative_s100_f0.pt` | qualitative binding assays | 100 | 0 | 0.35582 | `configs/predict/config_demo.py` |
| `prepibind_ms_s128_f3.pt` | mass-spectrometry eluted ligands | 128 | 3 | 0.12274 | `configs/predict/config_ms.py` |
| `prepibind_ic50_500_s128_f2.pt` | IC50, binder threshold < 500 nM | 128 | 2 | 0.46631 | `configs/predict/config_ic50_500.py` |
| `prepibind_ic50_1000_s42_f1.pt` | IC50, binder threshold < 1000 nM | 42 | 1 | 0.49456 | `configs/predict/config_ic50_1000.py` |

Each is 213.0 MiB, a one-key `{"model_state_dict": ...}` dict of 64 float32 tensors, 55,820,161
parameters. The seed and the fold are in the filename because they are part of what the file is.

All four rows are the val_loss argmin of their arm. **Not yet published:** the files exist locally
and nothing has been uploaded to HuggingFace; that is a human step, and the sequence is
`tools/release/upload_plan.md`.

## How these four were chosen, and why they are not the paper's numbers

Each arm was trained as **15 runs**: three seeds (42, 100, 128) by five cross-validation folds.
**Every number reported in the paper is a mean over those 15 runs.**

A released checkpoint is one single run out of the fifteen — the one with the **lowest validation
loss** in its arm, the value in the table above. **Validation loss is the selection criterion; the
`Val ROC-AUC` in the training logs is invalid — `train.py` defines `roc_auc_score(outputs, labels)`
and calls it with the arguments swapped, so 21.7 % of the logged values fall outside [0, 1] — and
it was never used for anything.** No test-set quantity took part in the choice either.

**The ms row changed on 2026-09-10, and only the ms row.** It was seed 100, fold 1 from
`250527/1_ms_re`, whose true val_loss is 0.13128. The corrected parse puts the arm's argmin at seed
128, fold 3 from `260829/ms`, 0.12274. That run was already one of the fifteen this arm averages:
`analysis/scoring/2_ms/per_run_results.csv` draws 9 of them from `250527/1_ms_re` and 6 from
`260829/ms` (seed 100 fold 4, and seed 128 folds 0-4). The two directories carry byte-identical
`config_global.json` and `config_esmc_small.py`, both log `Total samples: 77954` from the one
`data/dataset/ms_ql/train.csv`, and `260829`'s replicate of three seed-42 folds reproduces the
earlier root's validation losses exactly (0.14181, 0.14277, 0.14074). So this is a re-selection
inside the arm, on a criterion that did not change — not a new model, and not a new comparison.

The val_loss values above were re-derived on 2026-09-10 from `analysis/val_metrics.csv`, after a
regex in `analysis/tools/parse_train_logs.py` was fixed: it required an unsigned number for
`Val ROC-AUC`, so every epoch line carrying one of those negative values was dropped and the
recorded minimum was a minimum over a filtered subset. It could only ever come out too high, and on
46.7 % of runs it did. The training itself was never affected — `train.py` checkpoints on the true
minimum — which is why the epoch stored inside each of these four files (20, 27, 28, 28) matches the
corrected argmin and not the old one.

So a released checkpoint does not reproduce a paper number and is not meant to. It scores a little
above the mean on most metrics, because it was picked for a quantity correlated with test
performance — though not on all of them: F1 comes out slightly below the mean in the qualitative and
ic50_1000 arms. Both are below:

| arm | metric | paper (mean ± sd over 15 runs) | this checkpoint alone |
|---|---|---|---|
| qualitative | ROC AUC | 0.9193 ± 0.0028 | 0.9211 |
| | PR AUC | 0.9448 ± 0.0019 | 0.9460 |
| | F1 | 0.8586 ± 0.0037 | 0.8582 |
| | accuracy | 0.8375 ± 0.0038 | 0.8391 |
| | MCC | 0.6683 ± 0.0084 | 0.6736 |
| ms | ROC AUC | 0.9897 ± 0.0009 | 0.9911 |
| | PR AUC | 0.9865 ± 0.0012 | 0.9884 |
| | F1 | 0.9439 ± 0.0042 | 0.9502 |
| | accuracy | 0.9563 ± 0.0030 | 0.9612 |
| | MCC | 0.9083 ± 0.0063 | 0.9186 |
| ic50_500 | ROC AUC | 0.8375 ± 0.0039 | 0.8436 |
| | PR AUC | 0.7429 ± 0.0055 | 0.7520 |
| | F1 | 0.6764 ± 0.0146 | 0.6805 |
| | accuracy | 0.7685 ± 0.0035 | 0.7739 |
| | MCC | 0.4979 ± 0.0107 | 0.5067 |
| ic50_1000 | ROC AUC | 0.8347 ± 0.0044 | 0.8401 |
| | PR AUC | 0.8024 ± 0.0057 | 0.8093 |
| | F1 | 0.7399 ± 0.0083 | 0.7322 |
| | accuracy | 0.7554 ± 0.0039 | 0.7575 |
| | MCC | 0.5101 ± 0.0092 | 0.5110 |

Held-out test sets, never used for selection: 48,352 rows (qualitative), 33,490 (ms), 14,150 (both
IC50 arms). The per-run table these come from is `analysis/scoring/*/per_run_results.csv` in the
repository. Ignore its `val_auc` column, and the `Val ROC-AUC` in the training logs it comes from:
the `roc_auc_score` arguments are swapped and the values are meaningless. **Ignore its `val_loss`
column too** — it was written before the parser fix and is still the filtered minimum, 0.36951 for
the qualitative checkpoint and 0.14216 for the ms one. The validation losses in this card come from
`analysis/val_metrics.csv`, which was regenerated on 2026-09-10, and that is the only file to quote
them from.

## Training data

From the IEDB Export v3, split into arms by measurement type. The rebuild is
`pipeline/preprocess/` in the repository; the resulting CSVs are `data/dataset/` there.

| arm | directory | train rows | test rows | target column | positives (train) |
|---|---|---:|---:|---|---:|
| qualitative | `data/dataset/full/` | 112,871 | 48,352 | `Target` | 58.1 % |
| ms | `data/dataset/ms_ql/` | 77,954 | 33,490 | `Target` | 39.4 % |
| ic50_500 | `data/dataset/ic50/` | 33,004 | 14,150 | `Target_500` | 36.7 % |
| ic50_1000 | `data/dataset/ic50/` | 33,004 | 14,150 | `Target_1000` | 46.0 % |

The qualitative arm's training config names `dataset/full_bal/`, a directory since renamed to
`full`. Same CSVs.

Training: AdamW, lr 1e-5, betas (0.9, 0.999), eps 1e-8, weight decay 0.01, five-fold CV, best epoch
by validation loss (20, 27, 28, 28 for the four files above — the epoch each file stores, and the
corrected argmin of its run). Both sides were fed from precomputed
ESMC 300M embeddings, `emb_hla_esmc_small_0430.h5` for the HLA chains and the matching epitope
store.

## Architecture

`prepibind.model.plm_cat_mean_inf(hla_dim=960, epi_dim=960, head_div=64)`, 55,820,161 parameters:

- 2 self-attention blocks over the HLA chain-pair embeddings, 2 over the epitope embeddings,
  15 heads each (`960 // 64`)
- 1 joint self-attention block over the two concatenated token sequences
- masked mean pool, then `Linear(960, 480) -> GELU -> Dropout -> Linear(480, 1)`, one logit

Inputs are per-residue ESMC 300M embeddings, 960-dimensional. The head is not a language model and
never sees a raw sequence: the epitope is encoded at run time by ESMC 300M, the HLA alpha and beta
chains are read from a precomputed store.

## Input format

Two columns. The MHC field is the two chain names joined by an underscore. Both names must be keys
in the HLA mapping table the config points at, spelled exactly as that table spells them. The
datasets and the demo write **beta first, then alpha**; the order does not change the score, because
the head carries no positional encoding and pools over tokens, but following the convention keeps
inputs comparable with the published ones.

```csv
MHC,Epitope
HLA-DRB1*01:01_HLA-DRA*01:01,PKYVKQNTLKLATA
HLA-DPB1*03:01_HLA-DPA1*01:03,AAAARYPNVTIAAAA
H2-IAbB_H2-IAbA,AAAAPAAAATTAAPA
```

Output is the input with `Logits` and `Score` (sigmoid of the logit) appended.

Two limits worth knowing before feeding it anything:

- **Every training and test epitope is a 15-mer.** The arms were normalised to 15 residues. Other
  lengths run, since the encoder pads and masks, but they are outside the training distribution.
- **Coverage is 116 chains**: 98 human HLA class II chains and 18 mouse H2 chains, the ones the four
  arms actually use. The research embedding store carries 154 keys; the other 38 (20 BoLA, Mamu and
  SLA chains, plus 18 further human alleles) were curated alongside them, and no released model was
  trained to score them.

## Usage

```bash
git clone https://github.com/daylight-00/PREpiBind && cd PREpiBind
pip install --extra-index-url https://download.pytorch.org/whl/cu126 -e .

hf download daylight-00/prepibind           prepibind_qualitative_s100_f0.pt --local-dir models
hf download daylight-00/esmc-300m-2024-12   esmc_300m_2024_12_v0_fp16.pth    --local-dir models
hf download daylight-00/prepibind-embeddings emb_hla_esmc_small_0430.h5 --repo-type dataset --local-dir emb
```

The ESMC encoder is vendored into the package (`prepibind/esmc/`), so `torch` and
`huggingface_hub` are the only runtime dependencies for inference — no `pip install esm`.

```python
from prepibind.inference import load_config, main

cfg = load_config(
    "configs/predict/config_demo.py",                        # the qualitative model's config
    chkp_path="models/prepibind_qualitative_s100_f0.pt",     # this float32 checkpoint
    hla_emb_path="emb/emb_hla_esmc_small_0430.h5",           # full-length store
    hla_path="data/mhc_mapping/HLA2_IMGT_MSA_idx_edit.csv",  # mapping WITH the windows
    test_path="my_input.csv",
    out_path="outputs",
)
cfg["Test"]["precision"] = "as-trained"   # bfloat16 ESMC, float32 head. The default in that
                                          # config is "fp16", which is the demo's setting.
df = main(cfg)                            # writes outputs/prediction.csv, returns the DataFrame
```

Three things have to agree, and the code will tell you if they do not: a **full-length** store needs
the mapping table that carries the window as `sequence|start|end`
(`data/mhc_mapping/HLA2_IMGT_MSA_idx_edit.csv`), while the bundled demo store is already cut to the
window and needs `demo/data/mhc_mapping_demo.csv`, which has none. Applying a window twice raises.

The CLI does the same thing for the configs' own settings:

```bash
python -m prepibind.inference configs/predict/config_ms.py --chkp_path models/prepibind_ms_s128_f3.pt --plot
```

No GPU is required. Without CUDA, or on a pre-Ampere card, the runtime turns off flash-attn, falls
back to float32 and says so.

## Limitations and known caveats

- **One run, not the reported mean.** See above. For a benchmark comparison, use the 15-run means in
  the paper, not one checkpoint.
- **Class II only.** No class I allele is in the training data or in the embedding store.
- **The embedding store moved after training.** `emb_hla_esmc_small_0430.h5` as published in
  [`daylight-00/prepibind-embeddings`](https://huggingface.co/datasets/daylight-00/prepibind-embeddings)
  received the 2026-09-08 H2 chain-swap correction and differs from what these checkpoints were
  trained on in 3 of its 154 keys: `H2-IAdA` and `H2-IAdB`, which held each other's arrays, and
  `H2-IAg7A`, same shape and different values. The correction is disclosed, not retro-applied. All
  three are mouse H2 chains and all three are in the training data — 0.48 % to 0.66 % of the rows
  of each arm — so **predictions for H2-IAd and H2-IAg7 come from a model fit on the pre-fix
  arrays**. Every human HLA embedding is unchanged. The dataset card has the per-arm counts.
- **Training is not bit-reproducible** across GPU models, so a retrained checkpoint will differ in
  the last decimals. Inference on fixed inputs is deterministic.
- The models score binding, not immunogenicity. A high score is not a prediction that a peptide is
  presented in vivo, still less that it elicits a response.

## Licence

**MIT** for these weights and for the PREpiBind code, same as the repository.

The backbone is not ours, but it is also MIT. ESMC 300M moved to Chan Zuckerberg Biohub:
[`biohub/esmc-300m-2024-12`](https://huggingface.co/biohub/esmc-300m-2024-12) is ungated and its card
is tagged `mit` + `other`, and `esm` 3.4.0 — the release the vendored `prepibind/esmc/` source is
copied from — ships a plain MIT licence, "Copyright 2026 Chan Zuckerberg Biohub, Inc.". These
checkpoints were produced by training on ESMC 300M embeddings. Checked 2026-09-10; the repository's
`THIRD_PARTY_NOTICES.md` records the evidence and every other dependency.

Training data derives from the IEDB Export v3 (free to use, asks to be cited) and allele sequences
from IPD-IMGT/HLA and UniProt.

## Citation

```bibtex
@article{jang2026prepibind,
  title   = {PREpiBind: Protein Representation-integrated Epitope-MHC Class II Binding Prediction},
  author  = {Jang, David Hyunyoo and Kim, Dongwoo and Park, Byungho and Hwang, Untaek and Choi, Yoonjoo and Lee, Juyong},
  journal = {bioRxiv},
  year    = {2026}
}
```

The preprint is not posted yet; the DOI goes here when it is.
