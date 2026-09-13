# Terms for the data in this directory

The repository's root `LICENSE` is MIT and covers the **software**. It does not cover these files,
which are derived from third-party databases and carry those databases' terms. The same applies to
`demo/data/` and to `analysis/figures/data/`.

| Files | Source | Licence |
|---|---|---|
| `dataset/*/{train,test}{,_beta}.csv`, `unique_epitope_whole.csv` | IEDB, `mhc_ligand_full` export of 2025-04-21 (sha256 `7ce2af4d…70dc`) and the `tcell_full_v3` export of 2025-04-01 | **CC BY 4.0**. Filtered, relabelled and re-split — i.e. modified. Redistributed under the same licence. |
| `mhc_mapping/*.csv`, rows for HLA (116 of 154) | IPD-IMGT/HLA release 3.59.0, via its GitHub distribution [`ANHIG/IMGTHLA`](https://github.com/ANHIG/IMGTHLA) | **CC BY-NoDerivs**, per that repository's `LICENCE.md`. Modified here: gaps stripped, names collapsed, sequences windowed. Redistributing a modified version needs IPD's prior permission (`ipdsubs [at] anthonynolan [dot] org`), which **has not been obtained**. The notice asks for three citations; they are listed in `THIRD_PARTY_NOTICES.md`. |
| `mhc_mapping/*.csv`, rows for BoLA, SLA and Mamu (20 of 154) | IPD-MHC | **CC BY-NoDerivs**, same condition. |
| `mhc_mapping/*.csv`, rows for H2 (18 of 154) | UniProt (`P04228`, `P14434` and neighbours) | **CC BY 4.0**. Free to redistribute in modified form with attribution. |

Attribution owed, the citations to use, the exact snapshot identity and the unresolved IPD question
are all in [`../THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md). Cite the sources, and state
that the data was modified, when you reuse anything here.
