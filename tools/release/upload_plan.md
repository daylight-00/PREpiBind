# HuggingFace re-org — the command list

**Nothing in this file has been run.** It is the ordered, exact sequence for a human with write
access to `daylight-00` to execute. Every step that removes something is marked
**:warning: DESTRUCTIVE** and states what cannot be recovered afterwards.

> **All eight artifacts now exist and every number below was re-measured on 2026-09-10.** The last
> blocker, the **ms** arm's source, was decided that day: re-select within the arm's own paper runs,
> giving `prepibind_ms_s128_f3[_fp16].pt` from `260829/ms` (val_loss 0.12274) in place of the
> `250527` seed-100 fold-1 run (0.13128). `tools/release/README.md` step 1 records why the two are
> comparable. **Still not run, and still not runnable here:** every command below needs a token with
> write access to `daylight-00`, and nobody in this workstream has one or is authorised to use it.

State of the six repositories, re-read from the HF API on 2026-09-10 (unauthenticated):

| repo | type | exists today | after |
|---|---|---|---|
| `daylight-00/prepibind` | model | not visible — see below | created — 4 float32 checkpoints, 851.8 MiB |
| `daylight-00/prepibind-demo` | model | not visible — see below | created — 4 float16 checkpoints, 426.0 MiB |
| `daylight-00/prepibind-embeddings` | dataset | not visible — see below | created — 1 HDF5 store, 142.9 MiB |
| `daylight-00/esmc-300m-2024-12` | model | yes, 4 payload files + README + `.gitattributes` | edited **in place**: 2 `.h5` removed |
| `daylight-00/prepibind-esmc-300m` | model | yes, 4 checkpoints + README + `.gitattributes` | **deleted** |
| `daylight-00/emb_hla_esmc_small_0601_fp16` | dataset | yes, 7,282 `.npy` (1,003.2 MiB) + 2 | untouched (see step 6) |

**"Not visible" is not "404", and the difference matters.** An unauthenticated
`GET /api/models/daylight-00/prepibind` returns **HTTP 401** `{"error":"Invalid username or
password."}` — and so does `GET /api/models/daylight-00/zzz-does-not-exist-9f3a`, and so does a
repo under a namespace that does not exist at all. HF deliberately does not distinguish "absent"
from "private" to an anonymous caller, so no anonymous request can prove any of the three new names
is free. What *can* be said anonymously is that the public listing
`GET /api/models?author=daylight-00` returns exactly `prepibind-esmc-300m` and `esmc-300m-2024-12`,
and `GET /api/datasets?author=daylight-00` returns exactly `emb_hla_esmc_small_0601_fp16` — a
private repo would not appear there either. **Confirm with the token before step 2**, which costs
one command and is the only real check:

```bash
hf auth whoami
for r in prepibind prepibind-demo; do
  curl -s -o /dev/null -w "$r %{http_code}\n" \
    -H "Authorization: Bearer $(cat ~/.cache/huggingface/token)" \
    "https://huggingface.co/api/models/daylight-00/$r"
done   # 404 = free to create, 200 = it already exists, stop
```

`hf repos create` fails on an existing repo rather than overwriting it, so this is belt and braces
— but a 401 read as a 404 is exactly the mistake that makes someone add `--exist-ok` and then push
into a repo they did not expect to be there.

Order is not cosmetic: **everything is created and verified before anything is deleted.** A failure
part-way leaves the old artifacts still standing.

---

## Tooling

`huggingface-cli` is deprecated and no longer runs. The working CLI is `hf` (huggingface_hub 1.18.0),
which on this machine is:

```bash
export PATH=/home/hwjang/miniforge3/envs/venv/bin:$PATH
hf version            # version=1.18.0
hf auth login         # a token with WRITE scope on the daylight-00 namespace
hf auth whoami        # must print daylight-00 (or a user with write access to it)
```

`hf repo …` still works but prints a deprecation warning; the current spelling is `hf repos …`.
Uploads go over the HTTP API, which handles LFS itself — no `git lfs` and no hand-written
`.gitattributes` are needed. Repos created through the API get HF's default `.gitattributes`, which
already tracks `*.pt`, `*.pth` and `*.h5`.

---

## Step 0 — preconditions (read-only, run all of it)

```bash
cd /home/hwjang/project2/PREpiBind
PY=/home/hwjang/miniforge3/envs/venv/bin/python

# 1. all eight release files are byte-identical to a fresh derivation from the training
#    checkpoints. Run it one arm at a time: this login node's per-user memory cgroup killed a
#    single all-four process at ~6.4 GB RSS on 2026-09-10. ~15 s per arm.
for a in qualitative ms ic50_500 ic50_1000; do
  $PY tools/release/convert_checkpoints.py --check --arm $a --tmp-dir /home/hwjang/tmp_release
done                                                      # each must end in "ok"

# 1b. the two checks that --check cannot make: that this code produced the fp16 files already in
#     models/, and that every release file loads into the model the predict config builds.
for a in qualitative ms ic50_500 ic50_1000; do
  $PY tools/release/convert_checkpoints.py --check-legacy --arm $a --tmp-dir /home/hwjang/tmp_release
  $PY tools/release/convert_checkpoints.py --verify --arm $a
done   # --verify exits 1 on the ms arm until configs/predict/config_ms.py is fixed; see below

# 2. the embedding store is the one the dataset card describes
ls -l /home/hwjang/project/EMB/emb_hla_esmc_small_0430.h5  # 149,865,560 bytes
md5sum /home/hwjang/project/EMB/emb_hla_esmc_small_0430.h5 # 0454049f3d7122593dfff223811ecdba

# 3. the cards are current
ls tools/release/*.md
```

Also true before starting:

- The repository-side rename is already merged (`configs/predict/*.py`, `README.md`,
  `demo/run.ipynb`, `demo/run_colab.ipynb`, `THIRD_PARTY_NOTICES.md` name the new repo ids and the
  new filenames). Uploading first and editing the repo later leaves the README pointing at a repo
  whose files have different names.
- The `models/` directory holds all eight files, 1,339,858,440 B / 1,277.8 MiB (git-ignored; see
  `tools/release/README.md`), plus four superseded `prepi_esmc_small_*_fp16.pt` and four staged ms
  candidates in `models/ms_candidates/`. Neither of those last two groups is uploaded anywhere.
- **One blocker is outside this directory and is not fixed:** `configs/predict/config_ms.py` still
  sets `chkp_path` to `models/prepibind_ms_s100_f1_fp16.pt`, which does not exist — the ms arm
  re-selected to `prepibind_ms_s128_f3_fp16.pt`. Uploading is safe, but the demo for that arm is
  broken from a clean checkout until whoever owns `configs/` changes that one line.
  `convert_checkpoints.py --verify --arm ms` fails on exactly this and on nothing else.

---

## Step 1 — produce the artifacts (local, no network)

```bash
$PY tools/release/convert_checkpoints.py --write     # 8 files into models/, 1.34 GB
$PY tools/release/convert_checkpoints.py --check     # must end in "ok" (per --arm; see step 0)
$PY tools/release/convert_checkpoints.py --manifest  # the tables quoted below
```

`--write` refuses to touch a file that already exists; `--force` is the only way past it, and it
says which files it replaced. `--stage-ms` writes both ms candidates into `models/ms_candidates/`
under names that are not release names; that is where the 2026-09-10 comparison was made, and
nothing it writes is ever uploaded.

---

## Step 2 — CREATE `daylight-00/prepibind` (research tier, float32)

```bash
hf repos create daylight-00/prepibind --type model --public
hf upload daylight-00/prepibind tools/release/model_card_prepibind.md README.md \
    --commit-message "Model card"
```

Then the four checkpoints (213.0 MiB each, 851.8 MiB total). One command per file, so a failure
names the file it failed on:

```bash
hf upload daylight-00/prepibind models/prepibind_qualitative_s100_f0.pt prepibind_qualitative_s100_f0.pt
hf upload daylight-00/prepibind models/prepibind_ms_s128_f3.pt          prepibind_ms_s128_f3.pt
hf upload daylight-00/prepibind models/prepibind_ic50_500_s128_f2.pt    prepibind_ic50_500_s128_f2.pt
hf upload daylight-00/prepibind models/prepibind_ic50_1000_s42_f1.pt    prepibind_ic50_1000_s42_f1.pt
```

| file | bytes | MiB | md5 | on disk today |
|---|---:|---:|---|---|
| `prepibind_qualitative_s100_f0.pt` | 223,302,574 | 213.0 | `e0d105b190e03dcc54d889735bd65099` | yes |
| `prepibind_ms_s128_f3.pt` | 223,301,834 | 213.0 | `fd30ab20b3e096f9ee0cca8ff8a0a1a1` | yes |
| `prepibind_ic50_500_s128_f2.pt` | 223,302,370 | 213.0 | `c589a329ceaf3120752c5c78e3062cb8` | yes |
| `prepibind_ic50_1000_s42_f1.pt` | 223,302,370 | 213.0 | `708a3753ffae86beec884b31e62401fc` | yes |
| **total (4)** | **893,209,148** | **851.8** | | |

All four measured from disk with `stat` and `md5sum` on 2026-09-10. sha256 for each is in
`convert_checkpoints.py --manifest`, which also measures from disk rather than quoting.

An earlier version of this table carried an ms row of 223,301,834 B / md5
`d7390865bef9a5c5c55cdcac56d7a13d` — the *predicted* result of converting the old `250527` source
under a release name. The byte count happens to be unchanged, because `prepibind_ms_s100_f1` and
`prepibind_ms_s128_f3` are the same length and the size depends on the name; the md5 is completely
different, because the weights are. That is the trap this table is now free of: **never quote a hash
for a file that has not been built.**

---

## Step 3 — CREATE `daylight-00/prepibind-demo` (demo tier, float16)

```bash
hf repos create daylight-00/prepibind-demo --type model --public
hf upload daylight-00/prepibind-demo tools/release/model_card_prepibind_demo.md README.md \
    --commit-message "Model card"

hf upload daylight-00/prepibind-demo models/prepibind_qualitative_s100_f0_fp16.pt prepibind_qualitative_s100_f0_fp16.pt
hf upload daylight-00/prepibind-demo models/prepibind_ms_s128_f3_fp16.pt          prepibind_ms_s128_f3_fp16.pt
hf upload daylight-00/prepibind-demo models/prepibind_ic50_500_s128_f2_fp16.pt    prepibind_ic50_500_s128_f2_fp16.pt
hf upload daylight-00/prepibind-demo models/prepibind_ic50_1000_s42_f1_fp16.pt    prepibind_ic50_1000_s42_f1_fp16.pt
```

| file | bytes | MiB | md5 | on disk today |
|---|---:|---:|---|---|
| `prepibind_qualitative_s100_f0_fp16.pt` | 111,662,594 | 106.5 | `814b2c2be146d77d558c96e823a8fcfe` | yes |
| `prepibind_ms_s128_f3_fp16.pt` | 111,661,918 | 106.5 | `5c72435dfb42b3ec0af6c75c8eb83ef4` | yes |
| `prepibind_ic50_500_s128_f2_fp16.pt` | 111,662,390 | 106.5 | `3523cff0cd48e1fbe8ef4afd943d7860` | yes |
| `prepibind_ic50_1000_s42_f1_fp16.pt` | 111,662,390 | 106.5 | `4809f227098e5fb0f6417e951a4cc1bd` | yes |
| **total (4)** | **446,649,292** | **426.0** | | |

These are *not* the four files in `models/prepi_esmc_small_*_fp16.pt` renamed, and for two different
reasons. For three arms the tensors are identical and only the bytes differ, because `torch.save`
writes the output basename into the archive 68 times and the old name is still inside the old file.
For **ms** the tensors differ too, in all 64 of them: `prepi_esmc_small_ms_e5_s100_f1_fp16.pt` was
built from the superseded `250527` run. Upload what `convert_checkpoints.py` emitted, and never
`mv` an old file into a release name.

---

## Step 4 — CREATE `daylight-00/prepibind-embeddings` (dataset)

```bash
hf repos create daylight-00/prepibind-embeddings --type dataset --public
hf upload daylight-00/prepibind-embeddings tools/release/dataset_card_prepibind_embeddings.md \
    README.md --type dataset --commit-message "Dataset card"
hf upload daylight-00/prepibind-embeddings \
    /home/hwjang/project/EMB/emb_hla_esmc_small_0430.h5 emb_hla_esmc_small_0430.h5 --type dataset
```

| file | source | bytes | MiB |
|---|---|---:|---:|
| `emb_hla_esmc_small_0430.h5` | `/home/hwjang/project/EMB/emb_hla_esmc_small_0430.h5` | 149,865,560 | 142.9 |

154 datasets, float32, `(L, 960)`, 21 distinct `L` from 81 to 266. These are ESM C outputs, and ESM C
300M is now MIT under Chan Zuckerberg Biohub (`biohub/esmc-300m-2024-12`, checked 2026-09-10), so the
card carries `license: mit`.

The store received the 2026-09-08 H2 chain-swap correction and therefore differs from what the four
released checkpoints were trained on in 3 of its 154 keys. This is disclosed in the dataset card and
in both model cards, not retro-applied (`IMG/docs/decisions/h2-fix-is-not-retro-applied.md`).

---

## Step 5 — verify the three new repos before deleting anything (read-only)

```bash
for r in daylight-00/prepibind daylight-00/prepibind-demo; do
  curl -s "https://huggingface.co/api/models/$r?blobs=true" | $PY -c \
    "import sys,json;d=json.load(sys.stdin);[print(s['rfilename'],s.get('size'),(s.get('lfs') or {}).get('sha256','')) for s in d['siblings']]"
done
curl -s "https://huggingface.co/api/datasets/daylight-00/prepibind-embeddings?blobs=true" | $PY -c \
  "import sys,json;d=json.load(sys.stdin);[print(s['rfilename'],s.get('size')) for s in d['siblings']]"
```

Each `lfs.sha256` must equal the sha256 in `convert_checkpoints.py --manifest`. Then a real
round trip on one file per repo:

```bash
hf download daylight-00/prepibind prepibind_qualitative_s100_f0.pt --local-dir /tmp/hfcheck
md5sum /tmp/hfcheck/prepibind_qualitative_s100_f0.pt      # e0d105b190e03dcc54d889735bd65099
hf download daylight-00/prepibind-demo prepibind_qualitative_s100_f0_fp16.pt --local-dir /tmp/hfcheck
md5sum /tmp/hfcheck/prepibind_qualitative_s100_f0_fp16.pt # 814b2c2be146d77d558c96e823a8fcfe
rm -rf /tmp/hfcheck
```

Do the same round trip for `prepibind_ms_s128_f3.pt` (md5 `fd30ab20b3e096f9ee0cca8ff8a0a1a1`) and
`prepibind_ms_s128_f3_fp16.pt` (md5 `5c72435dfb42b3ec0af6c75c8eb83ef4`): that arm is the one whose
source changed on 2026-09-10, so it is the one worth confirming survived the upload intact.

Do not proceed to step 6 unless all of this matches.

---

## Step 6 — :warning: DESTRUCTIVE — trim `daylight-00/esmc-300m-2024-12` to the backbone

**Never `hf repos delete` this repository.** It holds `esmc_300m_2024_12_v0_bf16.pth`
(666,098,650 B, sha256 `bfc32e18b5c5430e2f349c19458169062aaa2b27fd543808cbb903bc586bb4cd`), and a
`find` over `/home/hwjang/project` and `/home/hwjang/project2` turns up **no local copy of it**.
Deleting the repo destroys that file. Edit in place.

Remote contents today:

| file | bytes | action |
|---|---:|---|
| `esmc_300m_2024_12_v0_fp16.pth` | 666,097,835 | keep (local copy matches: sha256 `d21625a4…73bf8b`) |
| `esmc_300m_2024_12_v0_bf16.pth` | 666,098,650 | **keep — no local copy exists** |
| `emb_hla_esmc_small_0601_fp16.h5` | 1,066,222,760 | delete |
| `emb_hla_esmc_small_light_0601_fp16.h5` | 19,765,392 | delete |
| `README.md`, `.gitattributes` | 152, 1519 | refresh / keep |

Precondition, and it needs a human answer, not a file check: **the webserver owner confirms the two
`.h5` are not read from here.** What is known is that
`daylight-00/emb_hla_esmc_small_0601_fp16` (dataset, 7,282 `.npy` shards, 1,003.2 MiB, last modified
2026-03-24) holds the same per-allele content, and `PREpiBind-web/configs/predict/config_demo.py`
reads `data/emb_hla_esmc_small_0601_fp16` with the web README documenting a download from that
dataset repo. So this is a deletion, not a migration — nothing has to be uploaded anywhere first.
Local copies of both files survive the deletion at
`/home/hwjang/project/PREpiBind/demo/data/emb_hla_esmc_small_{0601,light_0601}_fp16.h5`.

```bash
# :warning: DESTRUCTIVE
hf repos delete-files daylight-00/esmc-300m-2024-12 \
    "emb_hla_esmc_small_0601_fp16.h5" "emb_hla_esmc_small_light_0601_fp16.h5" \
    --commit-message "Backbone weights only: the 0601 HLA stores are webserver assets"
```

Then refresh the 152-byte README so the card says what the repo now is: backbone weights only, MIT,
pointing at upstream `biohub/esmc-300m-2024-12` and at `daylight-00/prepibind`. **The mirror's card
currently reads `other` / `cambrian-open-license` and is mislabelled** — upstream relicensed to MIT
and the old `EvolutionaryScale/` path redirects to `biohub/`. Not drafted here: it is upstream's
model, not ours, and the card should stay a thin pointer.

Reversible? The commit removes the files from `main`; the LFS blobs stay in history until someone
runs `HfApi().permanently_delete_lfs_files(...)`, and both files also exist on local disk. So this
step is recoverable. **Do not** run `permanently_delete_lfs_files` as part of this re-org.

---

## Step 7 — :warning: DESTRUCTIVE AND IRREVERSIBLE — delete `daylight-00/prepibind-esmc-300m`

This is the superseded release: four fp16 `.pth` from a *different* run set than the four decided
checkpoints (`e5_s128_f4`, `ms_e5_s100_f0`, `ic50_500_e5_s128_f4`, `ic50_1000_e5_s128_f1`).

**Correction, 2026-09-10: all four have a local copy, sha256-identical to the published blob.** An
earlier draft of this plan said three of them existed nowhere else, which would have made this step
a real data loss. It is not. Verified by `sha256sum` against the `lfs.sha256` the HF API reports,
and re-verified on 2026-09-10 — all four local files present, all four sizes and sha256 prefixes
matching the API to the byte:

| file | bytes | sha256 | local copy, byte-identical |
|---|---:|---|---|
| `prepi_esmc_small_e5_s128_f4_fp16.pth` | 111,666,105 | `520de798…c23427` | `/home/hwjang/project/PREpiBind/demo/models/prepi_esmc_small_e5_s128_f4_fp16.pth` |
| `prepi_esmc_small_ms_e5_s100_f0_fp16.pth` | 111,665,862 | `07cfe9d5…81b680` | `/home/hwjang/project/250619/fp16/prepi_esmc_small_ms_e5_s100_f0_fp16.pth` |
| `prepi_esmc_small_ic50_500_e5_s128_f4_fp16.pth` | 111,666,270 | `5ff97733…ae67f3` | `/home/hwjang/project/250619/fp16/prepi_esmc_small_ic50_500_e5_s128_f4_fp16.pt` |
| `prepi_esmc_small_ic50_1000_e5_s128_f1_fp16.pth` | 111,666,882 | `a6a7af7c…018aeb` | `/home/hwjang/project/250619/fp16/prepi_esmc_small_ic50_1000_e5_s128_f1_fp16.pt_fp16.pth` |

Two of those local names disagree with the published name — `.pt` for one, a doubled
`.pt_fp16.pth` suffix for another — and the bytes are identical anyway, which is consistent with a
rename after the fact: `torch.save` embeds the basename *stem*, `mv` does not touch it. Whoever
keeps these should not "fix" the names by re-saving.

So what step 7 destroys is the repository, its history and its download counter — irreversible, and
still not to be done casually — but no unique byte stream. Nothing in the repository or the
preprint depends on it, and the preprint is not posted, so no citation breaks.

Preconditions, all of them:

1. Steps 2–5 done and verified.
2. `grep -rn "prepibind-esmc-300m" .` in both `PREpiBind` and `PREpiBind-web` returns nothing.
3. Optional, and now genuinely optional given the table above — a second copy of the four files
   (426 MiB):

```bash
hf download daylight-00/prepibind-esmc-300m --local-dir /home/hwjang/project2/archive/hf_prepibind-esmc-300m
```

```bash
# :warning: DESTRUCTIVE, IRREVERSIBLE — the repo, its history and its download counter are gone
hf repos delete daylight-00/prepibind-esmc-300m --type model --yes
```

---

## Step 8 — after (read-only)

- `hf repos ls` — five repos under the namespace: three models (`prepibind`, `prepibind-demo`,
  `esmc-300m-2024-12`) and two datasets (`prepibind-embeddings`,
  `emb_hla_esmc_small_0601_fp16`), plus whatever else predates this. Before the re-org the public
  listing holds three: `prepibind-esmc-300m` and `esmc-300m-2024-12` as models,
  `emb_hla_esmc_small_0601_fp16` as a dataset.
- Open the three new cards in a browser: the licence badge comes from the YAML frontmatter, so
  `prepibind`/`prepibind-demo` must read MIT and `prepibind-embeddings` must read `other`.
- Re-run the demo end to end from a clean checkout: `demo/run.ipynb` cell 1 downloads from
  `daylight-00/prepibind-demo` and `daylight-00/esmc-300m-2024-12`; nothing may 404.
- **Outside this repository, and outside this workstream:** `PREpiBind-web/pages/4_about.py`
  lines 57–59 link to `huggingface.co/daylight00/…` — no hyphen, a user that does not exist. Those
  three links are already broken and the re-org would carry them forward. Whoever owns
  `PREpiBind-web` should fix them to `daylight-00/prepibind`, `daylight-00/prepibind-demo` and
  `daylight-00/esmc-300m-2024-12`.

## Provenance of every number above

Re-measured 2026-09-10; each row says how, so the next person re-runs the check instead of
trusting the table. Anything not listed here was not verified.

Everything in this table was re-checked on 2026-09-10, after the ms decision, against the
filesystem and against the live HF API.

| claim | checked by | result |
|---|---|---|
| `daylight-00/{prepibind, prepibind-demo, prepibind-embeddings}` absent | `curl` API + `?author=` listing | **inconclusive anonymously** — all three return 401, and so does the control `zzz-does-not-exist-9f3a`; all three absent from the public listing |
| public listing under `daylight-00` | `GET /api/models?author=` and `/api/datasets?author=` | exactly `prepibind-esmc-300m`, `esmc-300m-2024-12` (models) and `emb_hla_esmc_small_0601_fp16` (dataset) |
| `esmc-300m-2024-12` file sizes and both sha256 | `curl .../api/models/...?blobs=true` | all 4 payload files match, README 152 B, `.gitattributes` 1,519 B, lastModified 2025-06-27 |
| `esmc_300m_2024_12_v0_fp16.pth` has a local copy | `sha256sum` vs `lfs.sha256` | identical, `d21625a4…5373bf8b` |
| `esmc_300m_2024_12_v0_bf16.pth` has none | `find /home/hwjang/project{,2} -name '*bf16*.pth' -size +100M` | no hit — claim holds |
| `prepibind-esmc-300m` 4 files, sizes | HF API | match; README is 24 B, `.gitattributes` 1,519 B |
| all 4 of those have local copies | `sha256sum` vs `lfs.sha256` | 4/4 present and identical — see step 7, the old claim was wrong |
| `emb_hla_esmc_small_0601_fp16` = 7,282 `.npy`, 1,003.2 MiB | HF API sibling list | exactly 7,282 `.npy` (1,051,895,936 B) plus README 152 B and `.gitattributes` 2,504 B |
| `emb_hla_esmc_small_0430.h5` 149,865,560 B / 142.9 MiB | `ls -l`, `md5sum` | match; md5 `0454049f3d7122593dfff223811ecdba` |
| 154 datasets, float32, `(L, 960)`, 21 distinct `L`, 81-266 | `h5py` over the store | all four match |
| local copies of the two `0601` stores survive step 6 | `ls -l` | 1,066,222,760 and 19,765,392 B, both present |
| release file sizes / md5 / sha256 | `convert_checkpoints.py --check`, per arm | **8 of 8** byte-identical to a fresh derivation from their sources |
| this code produced the shipped fp16 files | `convert_checkpoints.py --check-legacy` | 4 of 4 `prepi_esmc_small_*_fp16.pt` reproduced byte for byte under their own basenames |
| every release file is a loadable model | `convert_checkpoints.py --verify` | 8 of 8: 64 tensors, 55,820,161 parameters, `strict=True` accepted, fp16 == `fp32.half()` bit for bit, no NaN/Inf |
| the ms source is the arm's argmin, epoch 27 | `analysis/val_metrics.csv` + the checkpoint's stored `epoch` | 0.12274 at epoch 27; the file stores 27 |
| `PREpiBind-web/pages/4_about.py` lines 57-59 | `grep -n daylight00` | lines 57, 58, 59 — correct |
| `configs/predict/*.py` name the new files | `grep chkp_path`, `--verify` | 3 resolve; **`config_ms.py` names `prepibind_ms_s100_f1_fp16.pt`, which does not exist** — outside this workstream's files, reported not fixed |

## What is deliberately not here

- No `permanently_delete_lfs_files`.
- No upload of the two `0601` stores anywhere: they are already published as a shard dataset.
- No change to `daylight-00/emb_hla_esmc_small_0601_fp16`. It belongs to the webserver.
- No tagging or versioning of the new repos. If the preprint needs a frozen pointer, add a git tag
  on HF after the DOI exists; that is a separate decision.
