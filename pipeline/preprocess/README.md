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
| — | `fetch_mhc_alignment.py`  | `mhc_sequences/{HLA2_IMGT,MHC2MSA}.csv` — the IPD-IMGT/HLA alignment, downloaded, **not shipped**. Stage 0 needs it |
| 0 | `0_mhc_sequences.ipynb`   | `work/mhc/` — sequence and window tables covering every allele in the source databases, about 7,000 rows |
| 1 | `1_iedb_to_draft.ipynb`   | `draft.csv` — class II, linear peptides, no mutants |
| 2 | `2_arm_qualitative.ipynb` | `work/full/` |
| 3 | `3_arm_ic50.ipynb`        | `work/ic50/` |
| 4 | `4_arm_ms_ql.ipynb`       | `work/ms_ql/` — MS positives, qualitative negatives |
| 5 | `5_arm_ms_ic.ipynb`       | `work/ms_ic/` — MS positives, IC50 negatives |

The order is a dependency order, not a preference: stage 4 filters against stage 2 and stage 5
against stage 3, because each MS arm draws its negatives from the arm it is paired with. Stages 2-5
read `draft.csv`, so `--from 2` resumes without re-reading the export.

Every stage writes by bare filename into its own directory under `$PREPIBIND_WORK` (default
`preprocess/work/`). Nothing in the chain writes into `data/`; compare the rebuild against the
shipped copies, or copy it over, by hand.

`data/mhc_mapping/` is not written by the notebooks either. Its three 154-row tables are stage 0's
~7,000 rows reduced to the 116 alleles the datasets use plus 38 curated non-human, non-murine
entries, with the H2 chain correction applied afterwards. `build_mhc_mapping.py` is that reduction:
`--check` reports it, `--write` performs it.

These are the notebooks the published datasets were built with. Their hardcoded paths were replaced
by the names in `paths.py` and nothing else was rewritten, which is why some of them still print
their intermediate tables.

## What it needs

- **Memory.** Stage 1 loads the whole export with a two-row header, which is the only expensive
  step; we run it in a 200 GB allocation and it finishes in about two minutes. Stages 2-5 take
  under 30 seconds each and fit in a few GB.
- **`PREPIBIND_WORK`** for the intermediates, if you do not want them under `preprocess/work/`.

## Reproducing

`verify_outputs.py` md5s 20 shipped files against `expected_checksums.csv`: the sixteen
`data/dataset/*/{train,test}{,_beta}.csv`, the three `data/mhc_mapping/` tables and
`data/unique_epitope_whole.csv`. Every one of those is a committed path no stage writes, so it
checks the checkout, not a rebuild — it passes on a fresh clone with no export in hand.

To check a rebuild, compare `work/<arm>/` against `data/dataset/<arm>/` yourself. Run it from the
2025-04-21 export: the zip member of the retained `mhc_ligand_full_single_file.zip` is stamped
2025-04-21 20:01:12 US/Pacific, and earlier revisions of this file said 2025-04-20, a date no
artifact supports. The `*_beta.csv` are not produced by the arm notebooks in publishable form;
`build_beta.py --check` reconstructs them and reports 4/8 byte-identical, 4/8 the same rows reached
another way.

One file here is an input rather than a product: `data/unique_epitope_whole.csv`, the epitope key
list the embedding stores were built against. It covers every epitope in the four arms plus 5,550
that were embedded and never trained on.

Stage 0's other input, the IPD-IMGT/HLA class II alignment, is **not in this repository**. IPD-IMGT/HLA
is CC BY-NoDerivs and asks to be linked to rather than mirrored, so:

```bash
python pipeline/preprocess/fetch_mhc_alignment.py          # ~4 MB from github.com/ANHIG/IMGTHLA
```

downloads release 3.59.0 — the one the paper used — and rebuilds both tables into `mhc_sequences/`,
where stage 0 expects them. `--check` verifies without writing. `MHC2MSA.csv` comes back
**byte-identical** to the copy this repository used to ship (sha256 `d492c29e…`); `HLA2_IMGT.csv`
returns every one of its 7,267 rows with the same residues, plus 16 two-field names that did not
exist in the older generation, and normalises the two gap characters to `*`.

The two files that do ship are the hand decisions behind it, which nothing upstream records:

| file | what it records |
|---|---|
| `filtered_manual.json`| the 296 cases where several four-field names collapsed to one two-field name, with the sequence that was chosen and how many entries backed it |
| *(removed)*           | `range_final.txt` held the hand alignment behind the peptide-binding windows. It carried IPD-derived sequence and is no longer shipped; the windows survive as `start_idx`/`end_idx` in `data/mhc_mapping/mhc_sources.csv` and in Supplementary Table S4 |

The windows are data, not a derivation: stage 0 locates `Sliced_Seq` inside `HLA_Seq`, and in the
surviving intermediates that lookup fails for most alleles because the two tables come from
different generations. `build_mhc_mapping.py --check` measures all three: membership (116 of the 154
published rows are reachable from the datasets, the other 38 are curated extras), sequences (113 of
the 116 reproduce exactly; three DQB1 alleles differ outside the window, so every sliced sequence
matches) and the windows, which do not reproduce. `fetch_mhc_alignment.py --check` re-cuts the
windows from the freshly downloaded alignment and reports how many still match.

A regenerated `draft.csv` may not be byte-identical to one made from a different download of the
same IEDB release; we have seen a species name differ (`Hepatovirus A` against
`Hepatovirus ahepa`). No column the arms keep is affected.
