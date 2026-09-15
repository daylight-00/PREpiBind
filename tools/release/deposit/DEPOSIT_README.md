# PREpiBind — data deposit

Companion data for **PREpiBind: Protein Representation-integrated Epitope-MHC Class II Binding
Prediction**.

Code: <https://github.com/daylight-00/PREpiBind>

This deposit holds the three files the repository needs and cannot carry itself. Everything else
the paper is built from is in the repository. **You do not need this deposit to reproduce the
figures and tables** — `make figures` runs from the repository alone in about a minute. You need it
to go one step further back, to re-derive the scoring tables from the raw model predictions, or to
rebuild the four datasets from the source database export.

## Files

| file | size | what it is |
|---|---:|---|
| `0_raw_260910.tar.zst` | 294 MiB | prediction snapshot: every model output the paper's numbers are computed from |
| `mhc_ligand_full_single_file.zip` | 240 MiB | the IEDB Export v3 the datasets were built from, exactly as IEDB served it |
| `draft.csv.tar.zst` | 12 MiB | the stage-1 intermediate, so you can skip the 7.7 GB step |
| `SHA256SUMS` | | checksums for the three files above |

## Verify first

```bash
sha256sum -c SHA256SUMS
```

All three must say `OK`. Then, per file:

```bash
zstd -t 0_raw_260910.tar.zst          # -> "1249218560 bytes"
unzip -t mhc_ligand_full_single_file.zip   # -> "No errors detected"
zstd -t draft.csv.tar.zst
```

## 1. `0_raw_260910.tar.zst` — the prediction snapshot

12,035 files, 1.16 GiB unpacked. One prediction CSV per (model, seed, fold, evaluation), plus the
test sets they are scored against and the two reference tools' predictions. Directory paths mirror
the scratch tree that produced them, so a file's provenance is its path.

```bash
mkdir -p /data/prepibind
tar -I zstd -xf 0_raw_260910.tar.zst -C /data/prepibind   # creates /data/prepibind/0_raw
export PREPIBIND_RAW_ROOT=/data/prepibind/0_raw
make scoring
```

`make scoring` rewrites `analysis/scoring/*/{per_run,seed_level,representation_level,...}_results.csv`.
Those files are tracked in git, so `git diff` after the run is the reproduction check. The
`val_loss` and `val_auc` columns in `per_run_results.csv` are joined in from the tracked
`analysis/val_metrics.csv`, not computed from the snapshot, so they reproduce whether or not the
snapshot is present. `val_auc` is a known-bad value carried only for inspection; the training code
computed it with swapped arguments and 21% of the values fall outside [0,1]. Do not use it.

Per-file md5 and provenance are in the repository at
`analysis/raw_manifest.csv`. To check the unpacked tree against it:

```bash
cd analysis && PREPIBIND_RAW_ROOT=/data/prepibind/0_raw tools/snapshot.sh verify
```

**Expected output is `manifest 12038 rows: 12035 ok, 3 missing, 0 mismatched`, and it exits 0.**
The three are `hum_ani_full.csv` under `250511/` and `250520/`. They are not missing from your copy
of the project: they ship inside the repository as `analysis/figures/data/figure_inputs.tar.zst`,
the manifest lists them under both `state=bundled` and `state=snapshot`, and `rawpath.at()` finds
them in the bundle with no snapshot present. Only `figS1_dataset_overlap.ipynb` reads them, and it
is a `make figures` step. Nothing in `make scoring` touches them.

Two further things the snapshot deliberately leaves out:

- Four HLA embedding `.h5` stores (1.3 GB), needed only by Supplementary Figure S2, which redraws
  instead from the committed `analysis/figures/umap_cache/*.npz`. These **are** in the manifest, as
  the four `state=external` rows.
- The per-fold ROC/PR diagnostic PNGs (8,850 files, 271 MB). No analysis reads them and they
  regenerate from the predictions plus the test sets. These are **not** in the manifest — it holds
  no PNG row — so unlike the stores, their provenance is not recorded anywhere. The 8,850 / 271 MB
  figures come from `analysis/tools/snapshot.sh`'s header comment and were not re-measured here.

## 2. `mhc_ligand_full_single_file.zip` — the IEDB export

The single-file MHC ligand export from the Immune Epitope Database, <https://www.iedb.org>, as
downloaded for this study and unmodified. It holds one member, `mhc_ligand_full.csv`,
7,745,252,872 bytes, timestamped 2025-04-22 12:01.

```
sha256 of the zip     7ce2af4d57a60c6a5f5fc474b5636fc6471479dd3f204170c3ef26760d6370dc
sha256 of the member  4d6d451023dbf93f3be6c4d44880147d2f0d294901776fa6f1df43f7c284ba52
```

```bash
unzip mhc_ligand_full_single_file.zip          # -> mhc_ligand_full.csv, 7.7 GB
export PREPIBIND_IEDB_EXPORT=$PWD/mhc_ligand_full.csv
make mhc-alignment                             # the IPD-IMGT/HLA alignment, fetched, not shipped
make datasets
python pipeline/preprocess/verify_outputs.py   # 20/20 md5 matches
```

Stage 1 loads the whole export with a two-row header. That is the only expensive step: it wants
roughly 200 GB of memory and takes about two minutes. Stages 0 and 2-5 take under 30 seconds each
and fit in a few GB. See `pipeline/preprocess/README.md`.

IEDB owns this data. It is redistributed here only so the exact snapshot behind the published
datasets stays retrievable; the live database has moved on since. Cite IEDB, not us, for the
underlying measurements.

## 3. `draft.csv.tar.zst` — the stage-1 intermediate

`draft.csv` is what stage 1 produces from the export: MHC class II rows, linear peptides, no
mutants. 566,494,795 bytes, 1,752,305 lines. Unpacking it lets you rebuild the four dataset arms
without ever downloading or loading the 7.7 GB export.

```bash
tar -I zstd -xf draft.csv.tar.zst -C pipeline/preprocess/work/   # -> work/draft.csv
python pipeline/preprocess/fetch_mhc_alignment.py                # the IPD-IMGT/HLA alignment, fetched, not shipped
python pipeline/preprocess/run_all.py --only 0                   # HLA sequences and windows
python pipeline/preprocess/run_all.py --from 2                   # the four arms
python pipeline/preprocess/verify_outputs.py                     # 20/20
```

Do not run `make datasets` on this path: it starts at stage 1 and would overwrite `work/draft.csv`
from an export you may not have.

A `draft.csv` regenerated from a different download of the same IEDB release is not guaranteed to be
byte-identical to this one. We have observed a source-organism name differing between two
downloads. No column the four arms keep is affected, and the twenty output checksums still match.

## What this deposit does not contain

| not here | where it is | what you lose without it |
|---|---|---|
| model checkpoints | HuggingFace `daylight-00/prepibind` (float32) and `daylight-00/prepibind-demo` (float16) | inference on new peptides |
| the research HLA embedding store `emb_hla_esmc_small_0430.h5` | HuggingFace `daylight-00/prepibind-embeddings` | `make demo-assets`, and the last step of `make verify` |
| ESMC 300M encoder weights | HuggingFace `daylight-00/esmc-300m-2024-12`, MIT | encoding epitopes at run time |
| the epitope embedding stores, about 21 GB | **not published** | re-scoring from checkpoints, and retraining |
| NetMHCIIpan-4.3 and MixMHC2pred-2.0 | their own distributors, under their own licences | re-running the two reference tools. Their predictions are already in the snapshot at `0_raw/260830/ref/`, and the tables derived from them are in the repository |
| the figures' own inputs | in the repository | nothing. `make figures` needs no download |

So: this deposit takes you from the published predictions back to every number in the paper, and
from the source database forward to the published datasets. It does not let you regenerate a model
output. That step needs the embedding stores, which are not published, and GPUs.

## Provenance

Every file was produced on the ABO Cluster (`abc`) and verified there on 2026-09-10. The repository
commit this deposit is matched to is recorded in `description.md` / the deposit landing page.

## Licence

The repository is MIT. The IEDB export is IEDB's, redistributed unmodified. The prediction snapshot
and `draft.csv` are outputs of this study.
