# Third-party notices

PREpiBind's own code is MIT (see `LICENSE`). One directory, `prepibind/esmc/`, is vendored
third-party source: also MIT, but under someone else's copyright, so it keeps its own licence file
next to it. This repository also builds on models, tools and databases that carry their own terms.
This file records what this repository actually contains or calls, and where each set of terms
lives. It is a pointer, not a substitute for reading them.

## Redistributed here

| What | Where | Terms |
|---|---|---|
| HLA and H2 sequences, and the domain windows derived from them | `data/mhc_mapping/`, `pipeline/preprocess/mhc_sequences/` | IPD-IMGT/HLA and UniProt. Both are free to use with attribution; cite them as the database releases ask. |
| Epitope–allele measurements | `data/dataset/` | Derived from the IEDB Export v3. IEDB is free to use and asks to be cited; the export itself is not redistributed here. |
| ESM C encoder source, vendored from `esm` 3.4.0, modified as each file's header records | `prepibind/esmc/`, licence at `prepibind/esmc/LICENSE-esm.md` | **MIT** — the same terms as the rest of this repository, but a different copyright holder: "Copyright 2026 Chan Zuckerberg Biohub, Inc.". That is why the licence file travels with the directory. `rotary.py` additionally carries EleutherAI/HuggingFace's Apache-2.0 header, a separate grant that came with the file upstream. |
| ESM C 300M HLA embeddings (demo set) | `demo/data/emb_hla_esmc_small_demo_fp16.h5` | Produced by running ESM C 300M. See ESM below. |

## Called, not redistributed

| Model or tool | Used for | Terms |
|---|---|---|
| **ESM C 300M** (weights) | epitope and HLA embeddings; the demo encodes epitopes at run time | Downloaded from the upstream release, not shipped here: [`biohub/esmc-300m-2024-12`](https://huggingface.co/biohub/esmc-300m-2024-12), ungated, card tagged `mit` + `other`. The old `EvolutionaryScale/esmc-300m-2024-12` redirects there. Details in the dated note below. |
| **ESM3 Small** (`esm3-small-2024-08`) | one of the compared representations, called over the hosted API (`pipeline/embeddings/esm/`) | The API this code calls has moved from Forge (`forge.evolutionaryscale.ai`) to `biohub.ai`; the [Forge API Terms of Use](https://www.evolutionaryscale.ai/policies/terms-of-use) still resolve and the platform now also states an [Acceptable Use Policy](https://biohub.org/acceptable-use-policy/). The ESM3 open weights, which this repository does not use, are no longer Cambrian *Non-Commercial*: [`biohub/esm3-sm-open-v1`](https://huggingface.co/biohub/esm3-sm-open-v1) is ungated and its card text says MIT. Details in the dated note below. |
| **Chai-1** (`chai_lab`) | one of the compared representations | Code Apache-2.0. The weights are released under Chai Discovery's own model terms — read the model card before any non-research use. |
| **Boltz** | one of the compared representations | MIT. |
| **AlphaFold 3** | one of the compared representations | Code Apache-2.0. The model parameters are *not* in that licence: they are obtained separately from Google DeepMind under their Model Parameters Terms of Use, which restricts redistribution. `pipeline/embeddings/af3/` builds the code; it does not ship parameters. |
| **NetMHCIIpan-4.3** | published-tool baseline | DTU Health Tech academic licence. Obtain it from DTU; only its predictions on our test sets are kept, under `analysis/scoring/0_ref/`. |
| **MixMHC2pred-2.0** | published-tool baseline | Academic licence from the Gfeller lab. Same arrangement as above. |
| **PyMOL** | the structure panel of Figure 1 | Only needed to regenerate that panel (`analysis/figures/render_7nzf_architecture_assets.py`). Schrödinger's licence for the incentive build, or the open-source build. |

## Released weights and stores

Published on HuggingFace under `daylight-00`:

| Repository | Contents | Terms |
|---|---|---|
| [`prepibind`](https://huggingface.co/daylight-00/prepibind) | the four research checkpoints, float32 | MIT, like this repository. |
| [`prepibind-demo`](https://huggingface.co/daylight-00/prepibind-demo) | the same four checkpoints in float16, for the demo | MIT, like this repository. |
| [`prepibind-embeddings`](https://huggingface.co/datasets/daylight-00/prepibind-embeddings) | `emb_hla_esmc_small_0430.h5`, the full-length float32 HLA store (154 alleles) | Produced by running ESM C 300M; the ESM terms above apply to it, and those are now MIT, which is what the draft dataset card in `tools/release/` carries. |
| [`esmc-300m-2024-12`](https://huggingface.co/daylight-00/esmc-300m-2024-12) | a copy of the ESM C 300M backbone weights | Upstream's, not ours — and upstream is now MIT. **This mirror's card still reads `other` / `cambrian-open-license`, so it is mislabelled**; see the dated note below. Nothing was changed on HuggingFace from here. |

The checkpoints and the embedding store were both produced with ESM C 300M; if you redistribute
anything derived from them, check the ESM terms above as well — they are now MIT, but read them
rather than trusting this sentence.

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
