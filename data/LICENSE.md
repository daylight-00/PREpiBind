# Terms for the data in this directory

The repository's root `LICENSE` is MIT and covers the **software**. It does not cover these files,
which are derived from third-party databases and carry those databases' terms. The same applies to
`demo/data/` and to `analysis/figures/data/`.

| Files | Source | Licence |
|---|---|---|
| `dataset/*/{train,test}{,_beta}.csv`, `unique_epitope_whole.csv` | IEDB, `mhc_ligand_full` export of 2025-04-21 (sha256 `7ce2af4d…70dc`) and the `tcell_full_v3` export of 2025-04-01 | **CC BY 4.0**. Filtered, relabelled and re-split — i.e. modified. Redistributed under the same licence. |
| `mhc_mapping/*.csv` HLA rows (116 of 134); `demo/data/mhc_mapping_demo.csv`; one member of `analysis/figures/data/figure_inputs.tar.zst` | IPD-IMGT/HLA release 3.59.0, via [`ANHIG/IMGTHLA`](https://github.com/ANHIG/IMGTHLA) | **CC BY-NoDerivs**, redistributed by permission of Anthony Nolan, 24 September 2026. Gap-free chains under two-field names with the peptide-binding window as coordinates; processed data, not an official IPD-IMGT/HLA release. The alignment itself is not redistributed. Conditions and citations: [`../THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md). |
| `mhc_mapping/*.csv` H2 rows (18 of 134) | UniProt (`P04228`, `P14434` and neighbours) | **CC BY 4.0**. Free to redistribute in modified form with attribution. |

Attribution owed, the citations to use and the exact snapshot identity are in
[`../THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md). Cite the sources, and state that the
data was modified, when you reuse anything here.
