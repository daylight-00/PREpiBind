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
parameters. md5, in table order: `e0d105b1…`, `fd30ab20…`, `c589a329…`, `708a3753…`.

All four are the val_loss argmin of their arm.

## How these four were chosen, and why they are not the paper's numbers

Each arm was trained as **15 runs**: three seeds (42, 100, 128) by five cross-validation folds, and
**every number reported in the paper is a mean over those 15**. A released checkpoint is one of the
fifteen — the one with the lowest validation loss, the value in the table above. No test-set
quantity took part in the choice. For a benchmark comparison, use the paper's means, not one
checkpoint.

Quote the validation losses from `analysis/val_metrics.csv` in the repository. The `val_loss` and
`val_auc` columns of `analysis/scoring/*/per_run_results.csv` are not usable for this: the first
predates a parser fix, and the second comes from a `roc_auc_score` call with its arguments swapped.

Held-out test sets, never used for selection: 48,352 rows (qualitative), 33,490 (ms), 14,150 (both
IC50 arms).

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

**MIT** for these weights and for the PREpiBind code. The backbone is not ours but is also MIT:
ESMC 300M is now [`biohub/esmc-300m-2024-12`](https://huggingface.co/biohub/esmc-300m-2024-12), and
`esm` 3.4.0 — the release the vendored `prepibind/esmc/` source comes from — ships a plain MIT
licence. Training data derives from IEDB (CC BY 4.0) and allele sequences from IPD-IMGT/HLA
(CC BY-NoDerivs) and UniProt (CC BY 4.0). The repository's `THIRD_PARTY_NOTICES.md` has the detail.

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
