# Third-party notices

PREpiBind's own code is MIT (see `LICENSE`). One directory, `prepibind/esmc/`, is vendored
third-party source: also MIT, but under someone else's copyright, so it keeps its own licence file
next to it. This repository also builds on models, tools and databases that carry their own terms.
This file records what this repository actually contains or calls, and where each set of terms
lives. It is a pointer, not a substitute for reading them.

**`LICENSE` covers the code, not the data.** The tables under `data/`, `demo/data/` and
`analysis/figures/data/` are derived from IEDB, IPD-IMGT/HLA, IPD-MHC and UniProt, and carry those
databases' terms — CC BY 4.0 for the first and last, CC BY-NoDerivs for the two IPD databases. MIT
neither describes nor can grant them. See the dated data-licence note below.

## Redistributed here

| What | Where | Terms |
|---|---|---|
| HLA sequences and the domain windows derived from them | `data/mhc_mapping/`, `demo/data/mhc_mapping_demo.csv`, `pipeline/preprocess/mhc_sequences/`, one member of `analysis/figures/data/figure_inputs.tar.zst` | IPD-IMGT/HLA (human) and IPD-MHC (BoLA, SLA, Mamu), both **CC BY-NoDerivs**. **Redistributing these tables needs IPD's prior permission, which has not been obtained** — see the dated note below. |
| H2 (murine) chain sequences, 18 rows of the same tables | `data/mhc_mapping/` | UniProt, **CC BY 4.0**. Free to redistribute in modified form with attribution. |
| Epitope–allele rows derived from IEDB — 774,446 rows in `data/dataset/`, plus the epitope key list, the demo input and three tables inside `figure_inputs.tar.zst` | `data/dataset/`, `data/unique_epitope_whole.csv`, `demo/data/dataset_demo.csv`, `analysis/figures/data/figure_inputs.tar.zst` | **CC BY 4.0**, the licence IEDB states for its data. These are *modified* IEDB data — filtered, relabelled and re-split. Attribution and snapshot identity below. |
| ESMC encoder source, vendored from `esm` 3.4.0, modified as each file's header records | `prepibind/esmc/`, licence at `prepibind/esmc/LICENSE-esm.md` | **MIT** — the same terms as the rest of this repository, but a different copyright holder: "Copyright 2026 Chan Zuckerberg Biohub, Inc.". That is why the licence file travels with the directory. `rotary.py` additionally carries EleutherAI/HuggingFace's Apache-2.0 header, a separate grant that came with the file upstream. |
| ESMC 300M HLA embeddings (demo set) | `demo/data/emb_hla_esmc_small_demo_fp16.h5` | Produced by running ESMC 300M on the `data/mhc_mapping/` sequences, so the ESM terms below and the IPD terms above both bear on it. |

## Called, not redistributed

| Model or tool | Used for | Terms |
|---|---|---|
| **ESMC 300M** (weights) | epitope and HLA embeddings; the demo encodes epitopes at run time | Downloaded from the upstream release, not shipped here: [`biohub/esmc-300m-2024-12`](https://huggingface.co/biohub/esmc-300m-2024-12), ungated, card tagged `mit` + `other`. The old `EvolutionaryScale/esmc-300m-2024-12` redirects there. Details in the dated note below. |
| **ESM3 Small** (`esm3-small-2024-08`) | one of the compared representations, called over the hosted API (`pipeline/embeddings/esm/`) | The API this code calls has moved from Forge (`forge.evolutionaryscale.ai`) to `biohub.ai`; the [Forge API Terms of Use](https://www.evolutionaryscale.ai/policies/terms-of-use) still resolve and the platform now also states an [Acceptable Use Policy](https://biohub.org/acceptable-use-policy/). The ESM3 open weights, which this repository does not use, are no longer Cambrian *Non-Commercial*: [`biohub/esm3-sm-open-v1`](https://huggingface.co/biohub/esm3-sm-open-v1) is ungated and its card text says MIT. Details in the dated note below. |
| **Chai-1** (`chai_lab`) | one of the compared representations | Code Apache-2.0. The weights are released under Chai Discovery's own model terms — read the model card before any non-research use. |
| **Boltz** | one of the compared representations | MIT. |
| **AlphaFold 3** | one of the compared representations | Code Apache-2.0. The model parameters are *not* in that licence: they are obtained separately from Google DeepMind under their Model Parameters Terms of Use, which restricts redistribution. `pipeline/embeddings/af3/` builds the code; it does not ship parameters. |
| **NetMHCIIpan-4.3g** | published-tool baseline | DTU Health Tech academic licence. Obtain it from DTU. Neither the tool nor its predictions are in this repository: `analysis/scoring/0_ref/` holds only the code that runs it plus an allele-name map, and the two prediction tables live in the separately deposited `0_raw` snapshot. |
| **MixMHC2pred-2.0.2** | published-tool baseline | Academic licence from the Gfeller lab. Same arrangement as above. |
| **PyMOL** | the structure panel of Figure 1 | Only needed to regenerate that panel (`analysis/figures/render_7nzf_architecture_assets.py`). Schrödinger's licence for the incentive build, or the open-source build. |

## Released weights and stores

Published on HuggingFace under `daylight-00`:

| Repository | Contents | Terms |
|---|---|---|
| [`prepibind`](https://huggingface.co/daylight-00/prepibind) | the four research checkpoints, float32 | MIT, like this repository. |
| [`prepibind-demo`](https://huggingface.co/daylight-00/prepibind-demo) | the same four checkpoints in float16, for the demo | MIT, like this repository. |
| [`prepibind-embeddings`](https://huggingface.co/datasets/daylight-00/prepibind-embeddings) | `emb_hla_esmc_small_0430.h5`, the full-length float32 HLA store (154 alleles) | Produced by running ESMC 300M; the ESM terms above apply to it, and those are now MIT, which is what the draft dataset card in `tools/release/` carries. |
| [`esmc-300m-2024-12`](https://huggingface.co/daylight-00/esmc-300m-2024-12) | a copy of the ESMC 300M backbone weights | Upstream's, not ours — and upstream is now MIT. **This mirror's card still reads `other` / `cambrian-open-license`, so it is mislabelled**; see the dated note below. Nothing was changed on HuggingFace from here. |

The checkpoints and the embedding store were both produced with ESMC 300M; if you redistribute
anything derived from them, check the ESM terms above as well — they are now MIT, but read them
rather than trusting this sentence.

### Data licence status, checked 2026-09-10

Licensor pages read directly on that date; this replaces the earlier claim that the IEDB export "is
not redistributed here", which was wrong — derived rows from it are shipped in four places.

**IEDB — CC BY 4.0, and that is enough.**

- [`iedb.org/database_export_v3.php`](https://www.iedb.org/database_export_v3.php), the page the
  export came from, and [`iedb.org/citation_v3.php`](https://www.iedb.org/citation_v3.php) both
  state: "All data is attributed to the publishing or submitting authors. This work is licensed
  under a Creative Commons Attribution 4.0 International License." Every iedb.org page repeats it
  machine-readably as JSON-LD `dcterms:license`.
- [`iedb.org/terms_of_use_v3.php`](https://www.iedb.org/terms_of_use_v3.php) adds that "NIAID places
  no restrictions on the use or distribution of the data contained within the IEDB", with the caveat
  that individual submitters may hold rights in their own submissions, which NIAID cannot assess.
- No IEDB page attaches conditions to *derived* datasets. The conditions are CC BY 4.0's own:
  attribution, and §3(a)(1)(B), **indicate that you modified the material**. There is no
  NoDerivatives and no ShareAlike, so filtered and relabelled tables may be redistributed.
- **The snapshot.** IEDB's CSV export carries no version label — no `Last-Modified`, and only the
  XML exports are archived — so a snapshot is identified by date plus checksum. The retained file is
  `mhc_ligand_full_single_file.zip`, 251,184,299 bytes, sha256
  `7ce2af4d57a60c6a5f5fc474b5636fc6471479dd3f204170c3ef26760d6370dc`; its single member
  `mhc_ligand_full.csv` (7,745,252,872 bytes) carries the timestamp **2025-04-21 20:01:12** US/Pacific,
  i.e. 2025-04-22 KST, and was downloaded 2025-04-24. `pipeline/preprocess/paths.py` and
  `pipeline/preprocess/README.md` said **2025-04-20** until 2026-09-13; no artifact supports that
  date and both now name the stamped one.
  **A second IEDB export is also used**, and only by `data/unique_epitope_whole.csv`: the T-cell
  assay table `tcell_full_v3.zip`, 43,216,977 bytes, sha256
  `09c87e9bed4f1d9d0fd594ccd674734ca3161d82d5b3cebeb28e85bbf39e86eb`, member `tcell_full_v3.csv`
  (1,287,618,749 bytes) dated 2025-04-01, from the same April 2025 download. It carries the same CC
  BY 4.0 terms. No dataset arm draws on it.
- **Attribution owed, and how it is met.** Cite IEDB — Vita R, Blazeska N, Marrama D, et al. *The
  Immune Epitope Database (IEDB): 2024 update.* Nucleic Acids Res. 2025;53(D1):D436–D443,
  doi:10.1093/nar/gkae1092 — and `www.iedb.org`. Stage 1 drops the PMID and submission-ID columns, so
  per-record attribution to submitting authors is carried by the named snapshot above rather than by
  a column, which is what CC BY §3(a)(2) permits.
- **What is *not* shipped.** No free-text IEDB field survives into any published file: reference,
  PMID, submission ID, authors, journal, title, assay comments, IRIs, host, disease and
  source-molecule columns are all absent from every published file. They are **not** dropped at
  stage 1 — `pipeline/preprocess/notebooks/1_iedb_to_draft.ipynb` carries most of them into the
  (gitignored) `draft.csv`; they are discarded when each arm notebook builds a fresh frame from the
  columns it needs. Verified against the published headers, not against the stage that was assumed
  to do it. **Three published columns are verbatim IEDB values**, and at scale: `Epi_Seq`
  (`Epitope > Name`), `HLA_Name_full` (`MHC Restriction > Name`), and `demo/data/dataset_demo.csv`'s
  `Epitope`. Measured against the retained export, not sampled: 774,446 + 319,821 + 48,352 published
  rows carry a byte-identical `Epitope > Name`, and 1,094,267 of them also carry a byte-identical
  `MHC Restriction > Name`; all 161,252 distinct (epitope, allele) pairs published anywhere occur as
  assay records in the export. `data/unique_epitope_whole.csv` is accounted for in full, but not from
  one source: 58,134 of its 63,146 rows are a verbatim `Epitope > Name` in the MHC-ligand export,
  3,498 come from the **IEDB T-cell assay export** named above, and the remaining 1,514 are generated
  here — four complete saturation-mutagenesis scans (15 positions x 19 substitutions each) and
  re-registered variants of published wild-types. Nothing in it is third-party data beyond IEDB.
  Every other column is derived here. The 566 MB `draft.csv`, which does retain IEDB metadata columns, is
  `.gitignore`d and is not public.
- **`data/` is CC BY 4.0, not MIT.** The repository's `LICENSE` is MIT and covers the code; it does
  not and cannot cover these rows, whose terms require attribution that MIT does not.

**IPD-IMGT/HLA and IPD-MHC — CC BY-NoDerivs; permission has not been asked.**

- **The sequences here came from the database's GitHub distribution,
  [`github.com/ANHIG/IMGTHLA`](https://github.com/ANHIG/IMGTHLA), not from a web download**, so that
  repository's own `LICENCE.md` is the operative notice. Read 2026-09-13, it states the same terms
  the IPD website does: "We have chosen to apply the Creative Commons Attribution-NoDerivs License
  to all copyrightable parts of our databases, which includes the sequence alignments. [...] We are
  strongly opposed to the mirroring of the data contained on our sites [...] **If you intend to
  distribute a modified version of our data, you must ask us for permission first, please contact
  ipdsubs [at] anthonynolan [dot] org**".
- **It asks for three citations, not one**, and all three are given in the paper:
  Barker DJ, Natarajan RHL, Cooper MA, Hopper SJF, Yates AD, Parham P, Marsh SGE, Robinson J,
  *The IPD-IMGT/HLA Database: recent developments in sequence submission*, Nucleic Acids Research
  (2026) 54(D1):D1152--D1158, doi:10.1093/nar/gkaf1218; Robinson J, Barker D, Marsh SGE,
  *25 years of the IPD-IMGT/HLA Database*, HLA (2024) 103(6):e15549; and Robinson J, Malik A,
  Parham P, Bodmer JG, Marsh SGE, *IMGT/HLA -- a sequence database for the human major
  histocompatibility complex*, Tissue Antigens (2000) 55:280--287.
- For the 20 BoLA, SLA and Mamu rows the operative page is
  [`ebi.ac.uk/ipd/licence/`](https://www.ebi.ac.uk/ipd/licence/), which says the same for IPD-MHC.
- What this repository ships against that: `pipeline/preprocess/mhc_sequences/MHC2MSA.csv` is a
  12,212-row **mirror** of the class II alignment, and `HLA2_IMGT.csv` and the three
  `data/mhc_mapping/` tables are **modified versions** of it — gap characters stripped, four-field
  names collapsed to two-field, sequences sliced to domain windows. Of the 154 published rows, 116
  are IPD-IMGT/HLA and 20 are IPD-MHC (BoLA, SLA, Mamu).
- **The release is 3.59.0** (2025-01-15). No artifact on disk records it, so it was recovered from
  the distribution's own `Allelelist_history.txt` on 2026-09-13: of the 110 releases it covers,
  **exactly one** contains all 12,212 allele names in `MHC2MSA.csv`. 3.58.0 is missing 187 of them
  and 3.60.0 has renamed or deleted 14, so the fit is unique.
- **No permission was requested and none has been granted.** Treat these six files as unresolved
  rather than cleared. NoDerivs restricts *distributing* adaptations, not using them, so the
  checkpoints, the training runs and every published number are unaffected. The two distributed
  embedding stores — `demo/data/emb_hla_esmc_small_demo_fp16.h5` and
  `daylight-00/prepibind-embeddings` — are a closer call, since their contents are per-residue
  representations of these sequences; include them in the same request.
- Cite in any case, in the form the licensor asks for at
  [`ebi.ac.uk/ipd/imgt/hla/about/citations/`](https://www.ebi.ac.uk/ipd/imgt/hla/about/citations/)
  ("For all citations please use"), read 2026-09-10:
  Barker DJ, Natarajan RHL, Cooper MA, Hopper SJF, Yates AD, Marsh SGE, Robinson J.
  *The IPD-IMGT/HLA Database: recent developments in sequence submission.*
  Nucleic Acids Research (2026) 54:D1152–D1158.
  (The licence page dates the same paper 2025 — advance access. The citations page is the operative
  one, and both list seven authors.)
  For IPD-MHC, [`ebi.ac.uk/ipd/licence/`](https://www.ebi.ac.uk/ipd/licence/) names Robinson J,
  Maccari G, Marsh SGE, et al. *KIR Nomenclature in non-human species*, Immunogenetics (2018) — but
  marks it "in preparation" and it does not resolve, so cite it as the licence page gives it and say
  where it came from rather than implying it is a locatable reference.

**UniProt — CC BY 4.0.** [`uniprot.org/help/license`](https://www.uniprot.org/help/license): "We
have chosen to apply the Creative Commons Attribution 4.0 International (CC BY 4.0) License to all
copyrightable parts of our databases." The 18 H2 rows (`P04228`, `P14434` and their neighbours) are
free to redistribute in modified form with attribution. This is a change from UniProt's older
CC BY-ND 3.0; the current page is the one that governs.

### ESM licence status, checked 2026-09-10

`esm` was relicensed and moved upstream between the first release draft and now. This records what
was actually resolved on that date, not what the older model cards say.

- `esm` 3.4.0's `LICENSE.md` is plain MIT, "Copyright 2026 Chan Zuckerberg Biohub, Inc.".
  `prepibind/esmc/LICENSE-esm.md` is that file byte for byte
  (md5 `27b8e07380db236f132b8f6252c739e8`, 1,095 bytes).
- `github.com/evolutionaryscale/esm` redirects to `github.com/Biohub/esm`, and
  `forge.evolutionaryscale.ai` redirects to `biohub.ai`. The Cambrian Open License page and the
  Forge terms page both still return 200 — they were not withdrawn, they simply no longer govern
  this code.
- `huggingface.co/EvolutionaryScale/esmc-300m-2024-12` redirects to `biohub/esmc-300m-2024-12`.
  That repository is public and **not gated** (`gated: false`), card `license: [mit, other]`,
  `license_link` pointing at upstream's `THIRD_PARTY_NOTICE.md` — which lists bundled *dependency*
  licences (flash-attn / PyTorch / xformers BSD, einops / jaxtyping / attrs MIT, lightning
  Apache-2.0), not a restrictive model licence. Nothing on the card mentions Cambrian.
- `EvolutionaryScale/esm3-sm-open-v1` likewise redirects to `biohub/esm3-sm-open-v1`, also
  ungated. Its README says "This repository is under a MIT license"; its card *metadata* carries
  no `license:` field at all, so the README is the only statement of terms there. Either way the
  Cambrian *Non-Commercial* licence this file used to cite for ESM3 is not what upstream now
  publishes.
- **Our mirror `daylight-00/esmc-300m-2024-12` is mislabelled**: the card is still
  `license: other`, `license_name: cambrian-open-license`, linking the Cambrian agreement, for
  weights whose upstream is MIT. Fixing that is a HuggingFace change and was not made from here.

## Python dependencies

Every runtime dependency and its resolved version is in `pixi.lock`, with licence metadata
available from the package index each was fetched from.
