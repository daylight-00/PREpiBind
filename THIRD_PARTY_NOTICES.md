# Third-party notices

PREpiBind's own code is MIT (`LICENSE`). This file records what else this repository contains or
calls, and whose terms those carry. It is a pointer, not a substitute for reading them.

**`LICENSE` covers the code, not the data.** The tables under `data/`, `demo/data/` and
`analysis/figures/data/` come from IEDB, IPD-IMGT/HLA, IPD-MHC and UniProt — CC BY 4.0 for the first
and last, CC BY-NoDerivs for the two IPD databases. Per-file terms: [`data/LICENSE.md`](data/LICENSE.md).

## Redistributed here

| What | Where | Terms |
|---|---|---|
| HLA chain sequences, full-length and gap-free, with the peptide-binding window carried alongside as coordinates. Only the demo table ships cut to that window | `data/mhc_mapping/`, `demo/data/mhc_mapping_demo.csv`, one member of `analysis/figures/data/figure_inputs.tar.zst` | IPD-IMGT/HLA release 3.59.0 (human) and IPD-MHC (BoLA, SLA, Mamu), both **CC BY-NoDerivs**, citations below. The alignment itself is **not** redistributed: `pipeline/preprocess/fetch_mhc_alignment.py` downloads it. |
| H2 (murine) chains, 18 rows of the same tables | `data/mhc_mapping/` | UniProt, **CC BY 4.0** |
| Epitope–allele rows derived from IEDB: 774,446 in `data/dataset/`, plus the epitope key list, the demo input and three tables inside the figure tarball | `data/dataset/`, `data/unique_epitope_whole.csv`, `demo/data/dataset_demo.csv`, `analysis/figures/data/figure_inputs.tar.zst` | **CC BY 4.0**. Modified IEDB data — filtered, relabelled, re-split |
| ESMC encoder source, vendored from `esm` 3.4.0 and modified as each file's header records | `prepibind/esmc/`, licence at `prepibind/esmc/LICENSE-esm.md` | **MIT**, but "Copyright 2026 Chan Zuckerberg Biohub, Inc." — which is why the licence travels with the directory. `rotary.py` also carries EleutherAI/HuggingFace's Apache-2.0 header from upstream |
| ESMC 300M HLA embeddings, demo set | `demo/data/emb_hla_esmc_small_demo_fp16.h5` | ESMC 300M run on the `data/mhc_mapping/` sequences, so the ESM and IPD terms both bear on it |

## Called, not redistributed

| Model or tool | Used for | Terms |
|---|---|---|
| **ESMC 300M** (weights) | epitope and HLA embeddings | Downloaded from [`biohub/esmc-300m-2024-12`](https://huggingface.co/biohub/esmc-300m-2024-12), ungated, card `mit` + `other` |
| **ESM3 Small** | one compared representation, over the hosted API | [Forge API terms](https://www.evolutionaryscale.ai/policies/terms-of-use); the platform is now `biohub.ai` with its own acceptable-use policy |
| **Chai-1** | one compared representation | Code Apache-2.0; weights under Chai Discovery's model terms — read the card before non-research use |
| **Boltz** | one compared representation | MIT |
| **AlphaFold 3** | one compared representation | Code Apache-2.0. Parameters are **not** in that licence: obtain them from Google DeepMind under their Model Parameters Terms of Use. `pipeline/embeddings/af3/` builds the code only |
| **NetMHCIIpan-4.3g** | published-tool baseline | DTU Health Tech academic licence. `analysis/scoring/0_ref/` holds only the runner and an allele-name map |
| **MixMHC2pred-2.0.2** | published-tool baseline | Academic licence from the Gfeller lab, same arrangement |
| **PyMOL** | the structure panel of Figure 1 | Only to regenerate that panel |

## Released weights and stores

On HuggingFace under `daylight-00`: [`prepibind`](https://huggingface.co/daylight-00/prepibind) and
[`prepibind-demo`](https://huggingface.co/daylight-00/prepibind-demo) (the four checkpoints, float32
and float16) are MIT like this repository;
[`prepibind-embeddings`](https://huggingface.co/datasets/daylight-00/prepibind-embeddings) is the
full-length HLA store, produced with ESMC 300M, so the ESM terms apply;
[`esmc-300m-2024-12`](https://huggingface.co/daylight-00/esmc-300m-2024-12) is a copy of upstream's
weights. **That mirror's card still reads `cambrian-open-license` for weights whose upstream is now
MIT, so it is mislabelled** — a HuggingFace-side fix, not made from here.

## Data sources: terms and attribution

**IEDB — CC BY 4.0.** Stated on the [export page](https://www.iedb.org/database_export_v3.php) and
the [citation page](https://www.iedb.org/citation_v3.php), and machine-readably as JSON-LD on every
iedb.org page. No NoDerivatives and no ShareAlike, so filtered and relabelled tables may be
redistributed; CC BY §3(a)(1)(B) requires saying that they were modified, which this file and the
paper do.

- **Snapshot.** The export carries no version label, so it is identified by date plus checksum:
  `mhc_ligand_full_single_file.zip`, 251,184,299 bytes, sha256 `7ce2af4d…0d6370dc`, whose member
  `mhc_ligand_full.csv` is stamped **2025-04-21** 20:01:12 US/Pacific. A second export,
  `tcell_full_v3.csv` of 2025-04-01 (sha256 `09c87e9b…f39e86eb`), feeds only
  `data/unique_epitope_whole.csv`. No dataset arm draws on it.
- **Attribution.** Cite IEDB — Vita R, Blazeska N, Marrama D, et al. *The Immune Epitope Database
  (IEDB): 2024 update.* Nucleic Acids Res. 2025;53(D1):D436–D443, doi:10.1093/nar/gkae1092 — and
  `www.iedb.org`. Per-record attribution to submitting authors is carried by the named snapshot
  rather than by a PMID column, which is what CC BY §3(a)(2) permits.
- **No free-text IEDB field is published.** Reference, PMID, submission ID, authors, journal, assay
  comments, host, disease and source-molecule columns are absent from every published file. Three
  published columns are verbatim IEDB values: `Epi_Seq`, `HLA_Name_full`, and the demo input's
  `Epitope`. Everything else is derived here.

**IPD-IMGT/HLA and IPD-MHC — CC BY-NoDerivs.**

- **Source.** The HLA sequences come from the database's own distribution,
  [`github.com/ANHIG/IMGTHLA`](https://github.com/ANHIG/IMGTHLA), **release 3.59.0** (2025-01-15).
  Its `LICENCE.md` governs them: CC BY-NoDerivs, and it asks that the data be linked to rather than
  mirrored. The alignment is therefore fetched, not shipped
  (`pipeline/preprocess/fetch_mhc_alignment.py`). What ships is 154 rows of gap-free chain
  sequence — 116 HLA, 18 H2 from UniProt, 20 BoLA/SLA/Mamu from IPD-MHC — uncut, with the
  peptide-binding window carried beside each row as coordinates; those window and collapse decisions
  are ours (`mhc_sequences/`). The cut is applied only in `demo/data/mhc_mapping_demo.csv`.
- **Citations, as `LICENCE.md` asks.** All three are cited in the paper: Barker DJ, Natarajan RHL,
  Cooper MA, Hopper SJF, Yates AD, Parham P, Marsh SGE, Robinson J, *The IPD-IMGT/HLA Database:
  recent developments in sequence submission*, Nucleic Acids Research (2026) 54(D1):D1152–D1158,
  doi:10.1093/nar/gkaf1218; Robinson J, Barker D, Marsh SGE, *25 years of the IPD-IMGT/HLA
  Database*, HLA (2024) 103(6):e15549; Robinson J, Malik A, Parham P, Bodmer JG, Marsh SGE,
  *IMGT/HLA — a sequence database for the human major histocompatibility complex*, Tissue Antigens
  (2000) 55:280–287.
- **IPD-MHC** (BoLA, SLA, Mamu rows) is covered by
  [`ebi.ac.uk/ipd/licence/`](https://www.ebi.ac.uk/ipd/licence/), same terms. It names Robinson J,
  Maccari G, Marsh SGE, et al. *KIR Nomenclature in non-human species*, Immunogenetics (2018);
  that reference is marked "in preparation" and does not resolve, so it is given as the page gives it.
- The demo embedding store and `daylight-00/prepibind-embeddings` are per-residue representations of
  these sequences and carry the same attribution.

**UniProt — CC BY 4.0** ([licence page](https://www.uniprot.org/help/license)). The 18 H2 rows
(`P04228`, `P14434` and neighbours) are free to redistribute in modified form with attribution.

**ESM, checked 2026-09-10.** `esm` 3.4.0's `LICENSE.md` is plain MIT, and
`prepibind/esmc/LICENSE-esm.md` is that file byte for byte. EvolutionaryScale's GitHub, Forge and
HuggingFace repositories now redirect to Biohub, where `esmc-300m-2024-12` is public and ungated
under MIT. The Cambrian licences this file once cited no longer govern this code.

## Python dependencies

Every runtime dependency and its resolved version is in `pixi.lock`, with licence metadata from the
index each was fetched from.
