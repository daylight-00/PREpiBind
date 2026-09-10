---
license: mit
tags:
  - biology
  - protein
  - immunology
  - mhc-class-ii
  - embeddings
  - esm
size_categories:
  - n<1K
---

# PREpiBind HLA embeddings — `emb_hla_esmc_small_0430.h5`

The full-length, float32 ESMC 300M embeddings of the MHC class II chains PREpiBind was trained and
evaluated on. One HDF5 file, 142.9 MiB, 154 datasets.

This is the store the research path reads. It is not the demo's store: the demo ships a smaller
float16 file cut to the peptide-binding window, inside the GitHub repository.

- Code and the pipeline that produced this: <https://github.com/daylight-00/PREpiBind>
- Models that read it: [`daylight-00/prepibind`](https://huggingface.co/daylight-00/prepibind)
- Backbone that produced it:
  [`daylight-00/esmc-300m-2024-12`](https://huggingface.co/daylight-00/esmc-300m-2024-12)

## Contents

| | |
|---|---|
| file | `emb_hla_esmc_small_0430.h5` |
| bytes | 149,865,560 (142.9 MiB) |
| datasets | 154, one per chain, keyed by bare allele name |
| dtype | float32 throughout |
| shape | `(L, 960)` — one 960-dimensional vector per residue |
| lengths | 21 distinct `L`, from 81 to 266 |
| attributes | none, on the file or on any dataset |

Keys are allele names exactly as `data/mhc_mapping/` spells them, alpha and beta chains as separate
entries:

| group | count | examples |
|---|---:|---|
| `HLA-DRB1` | 53 | `HLA-DRB1*01:01` |
| `HLA-DPB1` / `HLA-DQB1` / `HLA-DQA1` / `HLA-DPA1` | 18 / 17 / 13 / 4 | `HLA-DQA1*01:01` |
| `HLA-DRA` / `HLA-DRB3` / `HLA-DRB4` / `HLA-DRB5` | 1 / 4 / 3 / 3 | `HLA-DRA*01:01` |
| non-human: `H2` / `BoLA` / `Mamu` / `SLA` | 18 / 8 / 6 / 6 | `H2-IAbA`, `BoLA-DRB3*001:01` |

116 human HLA class II chains and 38 non-human chains curated alongside them.

The four released models were trained on **116 of these 154** keys, and it is not the same 116: the
arms use 98 of the human alleles and all 18 mouse H2 chains. The remaining 38, which are 18 further
human alleles plus the 20 BoLA, Mamu and SLA chains, are in the store because they were part of the
curation and are used by analyses of molecule coverage, not by any released checkpoint.

## How it was produced

ESMC 300M forward pass over each chain's full amino-acid sequence, per-residue hidden states
kept, `[CLS]` and `[EOS]` stripped, written as float32. The code is `pipeline/embeddings/esmc/` in
the repository; the sequences come from `data/mhc_mapping/`, which derives from IPD-IMGT/HLA and
UniProt. Nothing here is a measurement — it is a deterministic function of the sequences and the
backbone weights.

The peptide-binding window is **not** applied in this file. It is carried separately, in
`data/mhc_mapping/HLA2_IMGT_MSA_idx_edit.csv`, as `sequence|start|end`, and applied at load time.
Use that mapping table with this store. The demo's store is already cut and needs the mapping table
without windows; applying a window twice raises rather than silently returning the wrong residues.

## Disclosure: three keys changed after the released models were trained

On 2026-09-08 an H2 chain-swap error was corrected in the upstream sequences, and this file was
rebuilt for the affected chains. Exactly **3 of the 154 keys** differ from the arrays the four
released checkpoints were trained on:

| key | before | after |
|---|---|---|
| `H2-IAdA` | `(265, 960)` | `(256, 960)` |
| `H2-IAdB` | `(256, 960)` | `(265, 960)` |
| `H2-IAg7A` | `(256, 960)` | same shape, different values |

`H2-IAdA` and `H2-IAdB` had been written from each other's sequences. The governing decision is
*disclose, do not retro-apply*: re-running finished training on account of them would buy nothing at
the scale involved. The May 2025 training runs behind the released checkpoints read the **pre-fix**
arrays.

The scale, measured on the arms rather than asserted — rows whose alpha or beta chain is one of the
three changed keys:

| arm | training rows affected | test rows affected |
|---|---|---|
| qualitative | 547 of 112,871 (0.48 %) | 265 of 48,352 (0.55 %) |
| ms | 450 of 77,954 (0.58 %) | 215 of 33,490 (0.64 %) |
| ic50 | 218 of 33,004 (0.66 %) | 85 of 14,150 (0.60 %) |

Three consequences worth stating plainly:

- **This file carries no HDF5 attributes recording the fix**, so the difference is not detectable
  from the file itself. That is what this section is for. (The sibling `…_0329.h5` store does carry
  `h2_chain_fix*` attributes; this one does not.)
- Every human HLA embedding is unchanged. Anything that does not touch mouse H2 is unaffected.
- The 18 mouse H2 chains, these three included, **are** part of the four training arms. The affected
  rows are a real, if small, part of what the released checkpoints were fit on, which is why this is
  disclosed rather than dismissed.

If you re-train on this store you will get slightly different H2 behaviour from the released
checkpoints. If you run inference on human alleles, you will not.

## Usage

```bash
hf download daylight-00/prepibind-embeddings emb_hla_esmc_small_0430.h5 \
    --repo-type dataset --local-dir emb
```

Read it directly:

```python
import h5py
with h5py.File("emb/emb_hla_esmc_small_0430.h5", "r") as f:
    print(len(f))                       # 154
    emb = f["HLA-DRB1*01:01"][()]       # (L, 960) float32, full length
```

Or hand it to the model, which is what it is for:

```python
from prepibind.inference import load_config, main

cfg = load_config(
    "configs/predict/config_demo.py",
    chkp_path="models/prepibind_qualitative_s100_f0.pt",     # the float32 research checkpoint
    hla_emb_path="emb/emb_hla_esmc_small_0430.h5",           # this store
    hla_path="data/mhc_mapping/HLA2_IMGT_MSA_idx_edit.csv",  # the mapping WITH the windows
    test_path="my_input.csv",
    out_path="outputs",
)
cfg["Test"]["precision"] = "as-trained"
df = main(cfg)
```

The demo's 116-allele float16 store is derived from this file, and the derivation is checkable:

```bash
PREPIBIND_EMB_ROOT=$(pwd)/emb python demo/build_demo_assets.py hla-store --check
```

`PREPIBIND_EMB_ROOT` is the directory you downloaded this file into. The script's built-in default
path predates a repository move and no longer exists, so set the variable.

## Licence

**MIT.** These arrays are outputs of ESMC 300M. That model moved to Chan Zuckerberg Biohub —
[`biohub/esmc-300m-2024-12`](https://huggingface.co/biohub/esmc-300m-2024-12), ungated, card tagged
`mit` + `other` — so the Cambrian Open License Agreement this card previously cited no longer
governs it, and the same MIT grant that covers the PREpiBind code and weights applies here. Checked
2026-09-10; the repository's `THIRD_PARTY_NOTICES.md` records the evidence.

The underlying sequences come from IPD-IMGT/HLA and UniProt, both free to use with attribution;
cite them as those databases ask.

## Citation

```bibtex
@article{jang2026prepibind,
  title   = {PREpiBind: Protein Representation-integrated Epitope-MHC Class II Binding Prediction},
  author  = {Jang, David Hyunyoo and Kim, Dongwoo and Park, Byungho and Hwang, Untaek and Choi, Yoonjoo and Lee, Juyong},
  journal = {bioRxiv},
  year    = {2026}
}
```

Cite ESMC as EvolutionaryScale asks, as well: these embeddings are its output.
