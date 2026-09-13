# Preprocessing: IEDB export to the four dataset arms

Everything under `data/dataset/` is built here, from one external file: the IEDB Export v3
`mhc_ligand_full.csv` snapshot the paper used.

```bash
export PREPIBIND_IEDB_EXPORT=/path/to/mhc_ligand_full.csv   # 7.7 GB, unzipped
python pipeline/preprocess/run_all.py                       # all six stages
python pipeline/preprocess/verify_outputs.py                # md5 against expected_checksums.csv
```

`make datasets` runs the same thing; `make verify` runs every check, including the two derivations
that are reconstructions rather than producers (`build_beta.py`, `apply_h2_correction.py`).

## The stages

| | notebook | produces |
|---|---|---|
| 0 | `0_mhc_sequences.ipynb`   | `data/mhc_mapping/` — allele sequences and their domain windows |
| 1 | `1_iedb_to_draft.ipynb`   | `draft.csv` — class II, linear peptides, no mutants |
| 2 | `2_arm_qualitative.ipynb` | `data/dataset/full/` |
| 3 | `3_arm_ic50.ipynb`        | `data/dataset/ic50/` |
| 4 | `4_arm_ms_ql.ipynb`       | `data/dataset/ms_ql/` — MS positives, qualitative negatives |
| 5 | `5_arm_ms_ic.ipynb`       | `data/dataset/ms_ic/` — MS positives, IC50 negatives |

The order is a dependency order, not a preference: stage 4 filters against stage 2 and stage 5
against stage 3, because each MS arm draws its negatives from the arm it is paired with. Stages 2-5
read `draft.csv`, so `--from 2` resumes without re-reading the export.

These are the notebooks the published datasets were built with. Their hardcoded paths were replaced
by the names in `paths.py` and nothing else was rewritten, which is why some of them still print
their intermediate tables.

## What it needs

- **Memory.** Stage 1 loads the whole export with a two-row header, which is the only expensive
  step; we run it in a 200 GB allocation and it finishes in about two minutes. Stages 2-5 take
  under 30 seconds each and fit in a few GB.
- **`PREPIBIND_WORK`** for the intermediates, if you do not want them under `preprocess/work/`.

## Reproducing

`verify_outputs.py` checks 20 files: the sixteen `data/dataset/*/{train,test}{,_beta}.csv`, the
three `data/mhc_mapping/` tables and `data/unique_epitope_whole.csv`. All twenty match when the
chain is run from the 2025-04-21 export (the zip member of the retained
`mhc_ligand_full_single_file.zip` is stamped 2025-04-21 20:01:12 US/Pacific; earlier revisions of
this file said 2025-04-20, a date no artifact supports).

Two files here are inputs rather than products:

- `data/unique_epitope_whole.csv` — the epitope key list the embedding stores were built against.
  It covers every epitope in the four arms plus 5,550 that were embedded and never trained on.
- `mhc_sequences/HLA2_IMGT.csv` — the IPD-IMGT/HLA class II alignment stage 0 reads, one row per
  two-field allele name.

The other three files in `mhc_sequences/` are the record of how that alignment and the domain
windows were made, by hand, before this chain existed. No code reads them, and they are kept
because nothing else explains where those two things came from:

| file | what it records |
|---|---|
| `MHC2MSA.csv`         | the raw IPD-IMGT/HLA alignment, 12,212 rows under full four-field names |
| `filtered_manual.json`| the 296 cases where several four-field names collapsed to one two-field name, with the sequence that was chosen and how many entries backed it |
| `range_final.txt`     | the hand alignment behind the peptide-binding windows in `HLA2_IMGT_MSA_idx.csv` |

The windows are data, not a derivation: stage 0 locates `Sliced_Seq` inside `HLA_Seq`, and in the
surviving intermediates that lookup fails for most alleles because the two tables come from
different generations. `build_mhc_mapping.py --check` measures exactly what still reproduces.

A regenerated `draft.csv` may not be byte-identical to one made from a different download of the
same IEDB release; we have seen a species name differ (`Hepatovirus A` against
`Hepatovirus ahepa`). No column the arms keep is affected, and the twenty checksums still match.
