# Deposit description — draft, host-agnostic

Written so it can be pasted into Zenodo, figshare or an institutional repository without rewriting.
Placeholders in `<angle brackets>` are the only things that need filling.

---

## Title

PREpiBind: prediction snapshot and source data for "Protein Representation-integrated
Epitope-MHC Class II Binding Prediction"

## Authors

David Hyunyoo Jang, Dongwoo Kim, Byungho Park, Untaek Hwang, Yoonjoo Choi, Juyong Lee
<affiliations and ORCIDs as on the manuscript>

## Keywords

MHC class II; epitope prediction; peptide-MHC binding; protein language model; protein structure
prediction; benchmark; IEDB; reproducibility

## Licence

Files 1 and 3: <licence, e.g. CC BY 4.0>. File 2 is an unmodified redistribution of an Immune
Epitope Database export and remains subject to IEDB's terms.

## Related identifiers

| relation | identifier |
|---|---|
| is supplement to | the article, `<article DOI>` |
| is supplement to | the preprint, `<bioRxiv DOI>` |
| is supplemented by | software, <https://github.com/daylight-00/PREpiBind>, commit `<tag>` |
| is derived from | Immune Epitope Database, <https://www.iedb.org> |
| is derived from | IPD-IMGT/HLA, <https://www.ebi.ac.uk/ipd/imgt/hla/> |

---

## Description

PREpiBind is a peptide-MHC class II binding predictor that pairs a protein language model epitope
representation with several MHC representations, and a benchmark that compares eleven
representations under one training and evaluation protocol. This deposit holds the three data files
that the analysis code needs and that the source repository cannot carry: the raw model predictions
every reported number is computed from, the database export the training and test sets were built
from, and one intermediate that makes the second usable without a large-memory machine.

The repository at <https://github.com/daylight-00/PREpiBind> reproduces every figure and table in
the paper from tracked files alone, with no download and no GPU. This deposit extends that in two
directions. `0_raw_260910.tar.zst` contains 12,035 raw prediction files, one per model, seed, fold
and evaluation, together with the test sets they are scored against and the predictions of the two
reference tools; unpacking it and running `make scoring` re-derives every scoring table from the
predictions rather than reading the published ones. `mhc_ligand_full_single_file.zip` is the Immune
Epitope Database MHC ligand export used to build the four dataset arms, redistributed exactly as
served, so that the snapshot behind the published datasets stays retrievable after the live database
has moved on. `draft.csv.tar.zst` is the filtered intermediate produced from that export, included
because reading the 7.7 GB export needs about 200 GB of memory and the intermediate does not.

Directory paths inside the prediction snapshot mirror the compute tree that produced the files, so
each file's provenance is its path. A manifest in the repository records an md5, a size and the
consuming analysis for every file, including the ones deliberately left out and why.

What this deposit does not contain is stated in its README: model checkpoints, encoder weights and
the HLA embedding store are on HuggingFace, the epitope embedding stores are unpublished, and the
two reference tools are distributed under their own licences. Together, the repository and this
deposit take a reader from the source database forward to the published datasets, and from the
published predictions back to every number in the paper. They do not regenerate a model output.

Total size 545 MiB in three files. Checksums are published in the deposit and in the repository.
All files were verified on 2026-09-10.

---

## File manifest

| # | file | bytes | sha256 |
|---|---|---:|---|
| 1 | `0_raw_260910.tar.zst` | 308,022,032 | `a7aa11473824cee741fc1116569c5ab26fbe0c8ef00d3d67d9b5b56759149142` |
| 2 | `mhc_ligand_full_single_file.zip` | 251,184,299 | `7ce2af4d57a60c6a5f5fc474b5636fc6471479dd3f204170c3ef26760d6370dc` |
| 3 | `draft.csv.tar.zst` | 12,605,261 | `03871ed34d12c2202b7fa485b7522d14e1c4912f329f31e47c2596dfc377b364` |
| 4 | `README.md` | | usage, verification, and what is not here |
| 5 | `SHA256SUMS` | | checksums for 1-3 |

1. **`0_raw_260910.tar.zst`** — prediction snapshot. zstd-19 tar, 1,249,218,560 bytes unpacked,
   12,035 files in 1,512 directories under a single `0_raw/` root. Raw per-run prediction CSVs,
   their test sets, and the NetMHCIIpan-4.3 / MixMHC2pred-2.0 predictions. Consumed by `make
   scoring` through `PREPIBIND_RAW_ROOT`.
2. **`mhc_ligand_full_single_file.zip`** — Immune Epitope Database Export v3, single-file MHC ligand
   export, unmodified. One member, `mhc_ligand_full.csv`, 7,745,252,872 bytes, 4,883,622 lines,
   member timestamp 2025-04-22 12:01, member sha256
   `4d6d451023dbf93f3be6c4d44880147d2f0d294901776fa6f1df43f7c284ba52`. Consumed by `make datasets`
   through `PREPIBIND_IEDB_EXPORT`.
3. **`draft.csv.tar.zst`** — stage-0 intermediate `draft.csv`, 566,494,795 bytes, 1,752,305 lines:
   class II rows, linear peptides, mutants removed. Lets stages 2-5 run without the export.

---

## What differs between hosts, for this deposit

Input to a decision. Not a recommendation, and the host is explicitly still open.

**Size is not a discriminator.** 545 MiB clears every candidate comfortably. Zenodo's default
per-record limit is 50 GB, figshare's public-item allowance is effectively unbounded with a per-file
cap far above this, and institutional repositories that cap at 1-2 GB would still take it. Nothing
here needs a quota request, and nothing here should be chosen on size.

**Versioning and DOI granularity is the discriminator.** This snapshot has already been repacked
once: the previous archive predates one experiment directory and cannot reproduce the current
scoring tables. If review adds a run, the snapshot changes again, and the deposit needs a new
version whose old version stays resolvable, because the submitted manuscript will already cite a
DOI.

- Zenodo mints a version DOI per upload and a **concept DOI** that always resolves to the latest.
  Citing the concept DOI in the manuscript survives a re-upload; citing the version DOI pins the
  exact bytes. Both are available, which is the property that matters here.
- figshare versions an item in place and appends a version suffix to the DOI, with the base DOI
  resolving to the newest. Functionally similar; the older version stays reachable.
- Institutional repositories vary most on exactly this point. Some issue no DOI at all, some issue
  one per deposit with no versioning, in which case a corrected snapshot becomes a second,
  unlinked record. Confirm before choosing one.

**Code DOI versus data DOI.** Zenodo's GitHub integration mints a DOI for a tagged release
automatically, which would give the software and the data separate DOIs under one project. That is
arguably the right granularity: the code changes on a different schedule from the data. figshare and
most institutional repositories require the code to be uploaded as a file, which fixes it to the
data's version cycle.

**Third-party redistribution.** File 2 is 240 MB of IEDB data redistributed under our DOI. Zenodo
and figshare accept the uploader's licensing assertion; several institutional repositories review
third-party content before publishing. Whichever host is chosen, the underlying question of whether
we may redistribute this export at all is open and is upstream of the host choice.

**Longevity claims.** Zenodo commits to the lifetime of the CERN data centre. figshare has a
preservation arrangement through CLOCKSS. Institutional custodianship is tied to the institution.
For a five-year availability statement in the manuscript, any of the three is defensible; they
differ in what happens after that.
