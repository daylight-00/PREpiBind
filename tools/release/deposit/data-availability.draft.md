# Data Availability — draft paragraph

**This is a draft. The manuscript has not been edited.** It replaces the current
`\section{Data Availability}` in `IMG/paperwork/body/oup-authoring-template.tex`, which today is one
sentence naming GitHub plus a commented-out sentence naming IEDB and IPD-IMGT/HLA.

Applying it is a separate act by whoever owns that file. Two conventions apply and are not
pre-applied here: a content change of this size is wrapped in `\hy{}` per
`decisions/manuscript-edit-conventions.md`, and Data Availability is excluded from the body word
count, so its length does not compete with the 5,800-word target.

## Placeholders

| placeholder | filled from |
|---|---|
| `<DATA-DOI>` | the archival deposit for this study. Does not exist. See `README.md` §5 |
| `<CODE-DOI>` | a DOI for the tagged source release, if the code is deposited separately |
| `<IEDB-DATE>` | the export date. **Unresolved.** The file is stamped 2025-04-22; three places in the repository and the current commented-out sentence say 2025-04-20. See `README.md` §1.2 |
| `<IPD-RELEASE>` | the IPD-IMGT/HLA release. **Not recorded anywhere.** See `README.md` §5 item 4 |

## Draft

```latex
\section{Data Availability}

Source code, training and evaluation datasets, and the machine-readable result tables D01 to D12
are available at \url{https://github.com/daylight-00/PREpiBind} under the MIT License, and are
archived at \url{https://doi.org/<CODE-DOI>}.
Trained model checkpoints are deposited on HuggingFace at
\url{https://huggingface.co/daylight-00/prepibind}, with half-precision copies for the demonstration
notebooks at \url{https://huggingface.co/daylight-00/prepibind-demo}.
Pre-computed MHC class II embeddings are available at
\url{https://huggingface.co/datasets/daylight-00/prepibind-embeddings}.
The ESMC 300M base model weights are available from EvolutionaryScale under the Cambrian Open
License at \url{https://huggingface.co/daylight-00/esmc-300m-2024-12}.
The raw model predictions behind every reported value are archived at
\url{https://doi.org/<DATA-DOI>}, together with the database export the datasets were built from.
All figures and tables in this work can be regenerated from the repository alone; the deposit is
required only to re-derive the results tables from the raw predictions or to rebuild the datasets
from the export.
Peptide-MHC binding data were obtained from the Immune Epitope Database
(IEDB; \url{https://www.iedb.org}, export dated <IEDB-DATE>).
MHC allele sequences were retrieved from the IPD-IMGT/HLA database, release <IPD-RELEASE>
(\url{https://www.ebi.ac.uk/ipd/imgt/hla/}).
```

## If the deposit is not made before submission

Drop the two `\doi` sentences and keep the rest. The paragraph stays true: everything except the
raw predictions and the export is already public. Do not write "available on request"; the files
exist, are checksummed, and are 545 MiB.

## Checks before this is applied

- The four HuggingFace URLs currently return 401 to an anonymous request. Three of the four
  repositories do not exist yet. `tools/release/upload_plan.md` is the unrun sequence that creates
  them.
- `IMG/paperwork/body/webserver.tex` is a different manuscript and cites
  `daylight-00/prepibind-esmc-300m`, which `upload_plan.md` deletes, and
  `daylight-00/emb_hla_esmc_small_0601_fp16`, which it leaves alone. If the HuggingFace re-org
  happens, that paper's Data Availability section needs the same pass. Flagged, not changed.
- `<IPD-RELEASE>` has no source on disk. If it cannot be recovered, cite the database without a
  release rather than guess one.
- The current sentence's phrase "pre-trained models" is dropped, because the checkpoints are on
  HuggingFace and not in the repository, and `models/` is git-ignored.
