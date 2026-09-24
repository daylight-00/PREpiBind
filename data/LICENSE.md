# Terms for the data in this directory

The repository's root `LICENSE` is MIT and covers the **software**. It does not cover these files,
which are derived from third-party databases and carry those databases' terms. The same applies to
`demo/data/` and to `analysis/figures/data/`.

| Files | Source | Licence |
|---|---|---|
| `dataset/*/{train,test}{,_beta}.csv`, `unique_epitope_whole.csv` | IEDB, `mhc_ligand_full` export of 2025-04-21 (sha256 `7ce2af4d…70dc`) and the `tcell_full_v3` export of 2025-04-01 | **CC BY 4.0**. Filtered, relabelled and re-split — i.e. modified. Redistributed under the same licence. |
| `mhc_mapping/*.csv`, rows for HLA (116 of 134); `demo/data/mhc_mapping_demo.csv`; one member of `analysis/figures/data/figure_inputs.tar.zst` | IPD-IMGT/HLA release 3.59.0, via its GitHub distribution [`ANHIG/IMGTHLA`](https://github.com/ANHIG/IMGTHLA) | **CC BY-NoDerivs**, per that repository's `LICENCE.md`, and **redistributed here with permission granted by Anthony Nolan on 24 September 2026**, on the condition that Anthony Nolan be contacted to review the agreement should this work become commercial or otherwise make profit. What is here is the full-length gap-free chain under two-field names, with the peptide-binding window carried as coordinates (`start_idx`/`end_idx`, or `sequence|start|end`); `mhc_sources.csv` names the four-field allele behind each row. The alignment itself is not redistributed — `pipeline/preprocess/fetch_mhc_alignment.py` downloads it. These are processed data, not an official IPD-IMGT/HLA release. The notice asks for three citations; they are in `THIRD_PARTY_NOTICES.md`. |
| `mhc_mapping/*.csv`, rows for H2 (18 of 154) | UniProt (`P04228`, `P14434` and neighbours) | **CC BY 4.0**. Free to redistribute in modified form with attribution. |

Attribution owed, the citations to use, the exact snapshot identity and the unresolved IPD question
are all in [`../THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md). Cite the sources, and state
that the data was modified, when you reuse anything here.
