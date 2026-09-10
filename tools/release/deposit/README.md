# Archival deposit — prepared, not deposited

**Nothing here has been uploaded.** No DOI exists, no tag was made, and the host is not chosen.
This directory is the inventory, the verification log, the text, and one script that assembles the
deposit directory once someone decides where it goes.

```
tools/release/deposit/
  README.md                    this file: inventory, verification log, blockers
  inventory.csv                machine-readable: every candidate artifact, size, sha256
  stage_deposit.sh             assembles the deposit. Writes only inside its --out directory
  DEPOSIT_README.md            ships inside the deposit, for whoever downloads it
  description.md               deposit abstract, file manifest, and what differs between hosts
  data-availability.draft.md   draft manuscript paragraph. NOT applied to the manuscript
```

To assemble:

```bash
bash tools/release/deposit/stage_deposit.sh --out /some/scratch/prepibind-deposit
```

It copies two files, packs a third, writes `SHA256SUMS` and `README.md` into the output directory,
and verifies everything it wrote. It never deletes, never uploads, and touches nothing outside
`--out`. `--check` verifies the three sources and writes nothing at all.

**Both modes were run end to end on 2026-09-10**: `--check` in 34 s, the full assembly in 2 m 07 s,
producing 546 MiB in five files, all checksums re-verified and the packed `draft.csv` extracted and
compared byte for byte against its source. The test output directory was then deleted; no deposit
exists.

---

## 1. Inventory, verified on abc on 2026-09-10

Three files, **545.3 MiB** total.

| deposit name | bytes | MiB | sha256 (first 16) |
|---|---:|---:|---|
| `0_raw_260910.tar.zst` | 308,022,032 | 293.75 | `a7aa114738 24cee7` |
| `mhc_ligand_full_single_file.zip` | 251,184,299 | 239.55 | `7ce2af4d57 a60c6a` |
| `draft.csv.tar.zst` | 12,605,261 | 12.02 | `03871ed34d 12c220` |

Full checksums are in `inventory.csv`.

### 1.1 Prediction snapshot — `0_raw_260910.tar.zst`

Source: `/home/hwjang/project2/IMG/paperwork/plot_package/0_raw_260910.tar.zst`

| check | result |
|---|---|
| `zstd -t` | clean, 1,249,218,560 B uncompressed (1.163 GiB) |
| `sha256sum -c 0_raw_260910.tar.zst.sha256` | **OK**, still `a7aa1147...49142` |
| tar entries | 13,547 = 12,035 regular files + 1,512 directories |
| single top-level member | `0_raw/` |

Per-root file counts inside the archive: `250513` 6,500, `250524` 1,560, `260905` 1,620,
`250529` 878, `260830` 542, `250516` 480, `260829` 322, `250527` 128, `250714` 3, `250519` 2.
`260905/9_blosum_rerun` **is** present, and `260830/ref/pred_{netmhcpan,mixmhcpred}.csv` are too.

**File count against the tree it was packed from: 12,035 in the archive, 12,038 on disk.** The three
extras are

```
0_raw/250511/2_dataset/1_full_bal/hum_ani_full.csv
0_raw/250511/2_dataset/2_ic50/hum_ani_full.csv
0_raw/250520/1_dataset/3_ms_ql/hum_ani_full.csv
```

and they were written into the tree at 01:17 on 2026-09-10, **31 minutes after the archive was
packed at 00:46**. They are not a loss: the same three files, same md5, ship inside git as
`analysis/figures/data/figure_inputs.tar.zst`, `rawpath.at()` resolves them from there with no
snapshot, and only `figS1_dataset_overlap.ipynb` reads them — a `make figures` input, not a
`make scoring` one. Uncompressed sizes reconcile exactly: manifest snapshot rows total 1,211.6 MiB,
minus the 31.2 MiB of those three, plus tar headers, is the 1,249,218,560 B the archive holds.

One cosmetic consequence to expect, not a defect: `analysis/raw_manifest.csv` lists those three rels
**twice**, once `state=bundled` and once `state=snapshot`, so after unpacking the deposit
`analysis/tools/snapshot.sh verify` prints `manifest 12038 rows: 12035 ok, 3 missing, 0 mismatched`
and **exits 0**. `DEPOSIT_README.md` says so up front.

The archive does **not** contain `raw_manifest.csv` — unlike the older `0_raw.tar.zst`, which
`snapshot.sh pack` produced. The manifest is tracked in git at `analysis/raw_manifest.csv`, so
nothing is lost, but `snapshot.sh unpack` on this file yields `0_raw/` alone.

### 1.2 IEDB export — `mhc_ligand_full_single_file.zip`

Source: `/home/hwjang/project2/260901/0_snapshots/snapshotA_250422_mhc_ligand_full_single_file.zip`
Deposit it under the name IEDB served it as, `mhc_ligand_full_single_file.zip`.

| check | result |
|---|---|
| `unzip -t` | `No errors detected`, 31 s |
| members | exactly one: `mhc_ligand_full.csv`, 7,745,252,872 B, stamped **2025-04-22 12:01** |
| unzipped sha256 | `4d6d451023dbf93f3be6c4d44880147d2f0d294901776fa6f1df43f7c284ba52`, 4,883,622 lines |

Shipping the served zip as-is rather than repacking is the earlier conclusion and it holds: the
unzipped CSV is 7.7 GB, and a repack stops being the artifact IEDB distributed.

**There are two IEDB exports on abc and only one is the paper's.** The other,
`/home/hwjang/project/250408/3_filter/mhc_ligand_full_single_file.zip` (251,179,142 B, sha256
`480d3b74...b081b`), holds a `mhc_ligand_full.csv` stamped **2025-04-01 11:45**, 7,744,977,986 B,
4,883,427 lines. Distinguishing measurement, run today on all three files:

| file | rows containing `Hepatovirus` | of those, spelled `Hepatovirus ahepa` |
|---|---:|---:|
| 250408 export | 48 | **0** |
| 250422 export | 48 | **48** |
| `draft.csv` on disk | 2 | **2** |

`draft.csv` therefore came from the **2025-04-22** export, and cannot have come from the 2025-04-01
one. That is the same species-name difference
`notes/2026-09-10-prepibind-end-to-end-verified.md` reports between the staged and the regenerated
draft, now resolved to a direction. It also confirms the 2026-09-09 release-plan statement that
`250408/3_filter` is a different, earlier export.

Two things follow, and both are for a human:

- Deposit the **250422** zip. It is the export the published `data/dataset/` was built from.
- `pipeline/preprocess/paths.py:8`, `pipeline/preprocess/README.md` and the commented-out
  Data Availability sentence in `oup-authoring-template.tex:456` all say **2025/04/20**. Nothing on
  disk carries that date. Either IEDB labelled this export 2025-04-20 and the zip member is stamped
  two days later, or the date in the manuscript is wrong. Not settled here; see §5.

The 20/20 `verify_outputs.py` result recorded on 2026-09-10 was obtained from the **2025-04-01**
export, so the arms are insensitive to the difference between the two. That is a robustness fact
worth keeping. It is not a reason to deposit the wrong one.

### 1.3 Stage-0 output — `draft.csv.tar.zst`

Source: `/home/hwjang/project2/PREpiBind/pipeline/preprocess/input/draft.csv`

| | |
|---|---|
| bytes | 566,494,795 (540.25 MiB), 1,752,305 lines, two-row header |
| sha256 | `ce3b894c8c620920733514f6dd367e6a7af67573d70854c06918b827c316dde3` |
| md5 | `0cd74d5dcaaae24b53e965208e9bcffa` |
| mtime | 2026-09-10 01:35:54 |
| packed | `tar -cf - draft.csv \| zstd -19 -T0` = **12,605,261 B**, measured today, 2.2% |

12 MiB, so this is nearly free to include and it removes the only step that needs a 200 GB
allocation. `.tar.zst` rather than plain `.zst` keeps the deposit's two archives in one format;
the script has a single variable to change if that is not wanted.

`zstd -19 -T0` output depends on the thread count, so the script computes the checksum from the
file it actually wrote instead of trusting the number above. On this node (zstd 1.5.6, 48 threads)
it reproduces exactly: `03871ed34d12c2202b7fa485b7522d14e1c4912f329f31e47c2596dfc377b364`, and the
`draft.csv` extracted back out of it hashes to the source value above.

### 1.4 What else `make scoring` and `make datasets` need that is not in git

Derived from the `Makefile` and `analysis/README.md`, not guessed.

| target | external input | env var | in the deposit? |
|---|---|---|---|
| `figures` | none | — | n/a. `figure_inputs.tar.zst` and `umap_cache/*.npz` are tracked |
| `scoring` | unpacked prediction snapshot | `PREPIBIND_RAW_ROOT` | **yes**, item 1.1 |
| `datasets` | unzipped `mhc_ligand_full.csv`, 7.7 GB | `PREPIBIND_IEDB_EXPORT` | **yes**, item 1.2, plus 1.3 to skip stage 1 |
| `supplementary` | none | — | n/a. Reads only `analysis/scoring/*` and `analysis/figures/template.csv` |
| `verify` | research HLA store `emb_hla_esmc_small_0430.h5` | `PREPIBIND_EMB_ROOT` | **no** — HuggingFace `daylight-00/prepibind-embeddings` |
| `demo-assets` | same store | `PREPIBIND_EMB_ROOT` | **no** — same |

Two exclusions that are deliberate and should stay stated:

- `analysis/scoring/0_ref` is **not** in `make scoring`. Rescoring NetMHCIIpan-4.3 and
  MixMHC2pred-2.0 needs those tools installed under their own licences. Their predictions are in the
  snapshot at `0_raw/260830/ref/`, and their derived tables are tracked.
- The ~21 GB of epitope embedding stores (`~/project/EMB/*_0430*.h5`, 20 files) are in no deposit
  and no repository. Without them "re-derive our numbers" stops at "re-aggregate our predictions".
  Say so rather than let a reader infer otherwise.

### 1.5 The older `0_raw.tar.zst` — confirmed superseded, not deleted

`/home/hwjang/project2/IMG/paperwork/plot_package/0_raw.tar.zst`, 272,919,446 B, 2026-08-29 13:40,
sha256 `4f731f7e...61227`. Listed today: 10,758 entries, **9,552 files**, and **zero paths under
`260905/`**, so it predates `260905/9_blosum_rerun` exactly as claimed. It also carries
`raw_manifest.csv`, which the new archive does not.

It is 2,483 files short of the current snapshot and cannot reproduce the current scoring tables.
**Still on disk. Not deleted.** Delete it only after the new archive is deposited and the DOI
resolves.

---

## 2. Deposit README

`DEPOSIT_README.md` is written for a stranger with the GitHub repository and no context: what each
file is, which environment variable it becomes, which `make` target consumes it, the exact verify
commands, and a "what this deposit does not contain" section. It ships inside the deposit; the
staging script copies it in as `README.md`.

## 3. Description and manifest

`description.md` is the host-agnostic abstract, the file manifest, and one short section on what
actually differs between Zenodo, figshare and an institutional repository **for this deposit**.
That section is input to a decision. It does not make one.

## 4. Data Availability paragraph

`data-availability.draft.md` drafts the manuscript paragraph with `<DOI>` placeholders, naming the
GitHub repository, the four HuggingFace repositories, the D01-D12 tables, the snapshot, and the
IEDB and IPD-IMGT/HLA sources. **The manuscript is not edited.** Applying it is a separate,
deliberate act by whoever owns `oup-authoring-template.tex`.

---

## 5. What blocks depositing today

1. **No host, no account, no DOI.** Explicitly deferred. Everything below is what a decision needs.
2. **IEDB redistribution is unresolved.** Redistributing a 251 MB IEDB export under our own DOI is a
   licensing question nobody in this workstream has answered, and it was already on the list of six
   decisions with hwjang. If the answer is no, the deposit ships `draft.csv.tar.zst` plus the
   export's checksums and a download URL, and `make datasets` stage 1 stops being reproducible from
   the deposit alone. The rest is unaffected.
3. **Which export is "2025-04-20".** Measured above: the paper's export is stamped 2025-04-22 and
   was downloaded 2025-04-24. Three files in the public repository and one commented-out manuscript
   sentence say 2025-04-20. Somebody has to say which is right before the deposit's description
   states a date.
4. **No IPD-IMGT/HLA release number anywhere.** `pipeline/preprocess/mhc_sequences/HLA2_IMGT.csv`
   and `MHC2MSA.csv` (12,212 rows) carry no version, and neither do the READMEs. The Data
   Availability paragraph currently cites the database without a release. A release number would
   make `data/mhc_mapping/` reproducible; without one it is take-it-as-given.
5. **The HuggingFace repositories do not exist yet.** `tools/release/upload_plan.md` is unrun and
   needs a write token nobody here has. The Data Availability paragraph names four HF URLs that
   currently 401. Deposit and HF upload should land together, or the paragraph names dead links.
6. **Nothing is tagged.** A deposit that says "reproduces commit X" needs X to be a tag, not a
   moving `main`. `origin/main` is `5fe781b` as of today; the working tree has since moved.

None of these blocks *preparing*. All of item 1's artifacts are verified and the script runs today.

## 6. Deliberately not decided here

- Whether the four released checkpoints should also be in the archival deposit. HuggingFace is a
  host, not an archive, and a DOI that covers the data but not the weights is a common gap. Cost is
  about 1.25 GiB. Nobody has asked for this; raising it, not answering it.
- Whether `banner.png` and other repository content belong in a source-code archive alongside the
  data, or whether the code gets its own DOI through the GitHub-Zenodo integration.
