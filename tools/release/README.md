# `tools/release/` — preparing the HuggingFace re-org

Everything needed to publish the four released models, and nothing that publishes them. No command
in this directory touches HuggingFace: `convert_checkpoints.py` reads training checkpoints and
writes local files, and `upload_plan.md` is a list a human runs.

**All four arms are settled and all eight files are built.** The last open question, the **ms**
arm's source, was decided by hwjang on 2026-09-10: re-select within the arm's own paper runs, which
resolves to seed 128 fold 3 from `260829/ms` (val_loss 0.12274) in place of seed 100 fold 1 from
`250527/1_ms_re` (0.13128). Step 1 below records what makes the two comparable. Nothing here has
been uploaded, and nothing in this directory can upload it.

| file | what it is |
|---|---|
| `convert_checkpoints.py` | training checkpoints -> the release files, with `--check`, `--check-legacy`, `--verify` |
| `upload_plan.md` | the ordered `hf` command list, destructive steps marked |
| `model_card_prepibind.md` | card for `daylight-00/prepibind` (float32) |
| `model_card_prepibind_demo.md` | card for `daylight-00/prepibind-demo` (float16) |
| `dataset_card_prepibind_embeddings.md` | card for `daylight-00/prepibind-embeddings` |

The cards are uploaded **as `README.md`** into their repositories; they are kept here under
descriptive names so that all three can live in one directory.

## No `.gitignore` change was needed

The outputs go to **`models/`**, which `.gitignore` already excludes: the bare `models` rule matches
that directory at any depth, and `*.pth` / `*.h5` are excluded on top of it. Nothing in
`tools/release/` needed a new ignore rule, and none was added — worth saying explicitly, since
several people are editing this repository at once.

`models/` rather than a new `tools/release/out/` for a second reason: `configs/predict/*.py` load
`models/prepibind_<arm>_<seed>_f<fold>_fp16.pt`, so the demo tier lands exactly where the predict
configs and the notebooks expect it, and `python -m prepibind.inference configs/predict/config_demo.py`
works straight after `--write` with nothing to copy. The float32 tier lands beside it under names
that do not collide (no `_fp16` suffix).

**All eight exist today.** Measured from disk on 2026-09-10:

| tier | files | bytes | MiB |
|---|---:|---:|---:|
| research (float32) | 4 | 893,209,148 | 851.8 |
| demo (float16) | 4 | 446,649,292 | 426.0 |
| both tiers | 8 | 1,339,858,440 | 1,277.8 |
| `models/prepi_esmc_small_*_fp16.pt` (superseded, kept for now) | 4 | 446,651,260 | 426.0 |
| `models/ms_candidates/` (the 2026-09-10 comparison, never a release) | 4 | 669,932,512 | 638.9 |

Each research file is 213.0 MiB and each demo file 106.5 MiB; the small spread inside a tier is
basename length, not content — see step 3.

One consequence to hand to whoever owns `configs/`, and it is now a live bug rather than a pending
one: **`configs/predict/config_ms.py` still names `models/prepibind_ms_s100_f1_fp16.pt`**, a file
that does not exist and never will — the ms arm re-selected to `prepibind_ms_s128_f3_fp16.pt`. The
other three predict configs resolve to the right file.
`convert_checkpoints.py --verify` checks all four and fails on this one. `configs/` is outside this
directory's scope, so this is reported, not edited.

## Order of operations

Each step states what has to be true **before** it. None of them is reversible by the step after it.

### 1. Each source has to be the val_loss argmin of its arm

The source of truth is **`analysis/val_metrics.csv`**, regenerated 2026-09-10 (md5
`123fe7686e8ed9597cfaa04af3c5484e`, 5,150 runs) — not
`analysis/scoring/*/per_run_results.csv`, which is a test-set table.

| arm | seed | fold | val_loss | epoch | source checkpoint |
|---|---|---|---:|---:|---|
| qualitative | 100 | 0 | 0.35582 | 20 | `250513/1_bulk/2/models/e5_s100/esmc_small_fold0-best.pt` |
| ms | 128 | 3 | 0.12274 | 27 | `260829/ms/models_ql/e5_s128/esmc_small_fold3-best.pt` |
| ic50_500 | 128 | 2 | 0.46631 | 28 | `250516/7_ic50_etc/models_500/e5_s128/esmc_small_fold2-best.pt` |
| ic50_1000 | 42 | 1 | 0.49456 | 28 | `250516/7_ic50_etc/models_1000/e5_s42/esmc_small_fold1-best.pt` |

(under `/home/hwjang/project/`, and hard-coded in `ARMS` in `convert_checkpoints.py`.)

**These val_loss values changed on 2026-09-10 and the files did not.** `parse_train_logs.py`
matched `Val ROC-AUC: [\d.]+`, with no leading minus. The logged Val ROC-AUC is the corrupt
swapped-argument value and is negative on 21.7 % of epoch lines, so those lines were dropped whole
and the recorded per-run val_loss was a minimum over a filtered subset — necessarily too high, and
too high on 46.7 % of all runs. Corrected: qualitative 0.36951 -> 0.35582, ms 0.13929 -> 0.13128
for the previously released run and 0.12274 for the run that actually wins the arm; both IC50 arms
were already right. **The argmin *run* changed only for ms**, and that is the one selection this
correction moved.

The independent check that the new numbers are the right ones: `prepibind/train.py` saves
`-best.pt` at the true minimum, so the epoch inside the file is the true argmin epoch whatever the
log parser did afterwards. The four sources store 20, 27, 28, 28 — the corrected argmins to the
epoch. The old table said 23, 16, 28, 28 and agreed with none of the first two.

Three traps this encodes, each of which will silently mislabel a released file:

- **The seed is the one in the directory name `e5_s<seed>`.** `config_global.json` in all four trees
  says `seed: 100` and `lr: 1e-4`; `run.sh` overrides both on the command line, and the saved
  optimizer `param_group` confirms `lr=1e-05`. Reading the seed from the JSON mislabels three of the
  four checkpoints as `s100`.
- **`val_auc` is corrupt**, both in `per_run_results.csv` and in the `Val ROC-AUC` of the training
  logs it derives from — the `roc_auc_score` arguments are swapped, so the column holds things like
  `42.46` and `0.0027`. The selection criterion is `val_loss`, and nothing else ever was.
- **Never compare a val_loss you did not re-parse.** Any figure quoted from before 2026-09-10 is a
  filtered minimum and is not comparable with one from `val_metrics.csv`.

#### The ms arm: decided 2026-09-10, re-selected within the arm

The corrected argmin of the ms arm was not the released file:

| candidate | run | val_loss | stored epoch | outcome |
|---|---|---:|---:|---|
| previously released | `250527/1_ms_re/models_ql/e5_s100/esmc_small_fold1-best.pt` | 0.13128 | 20 | superseded |
| argmin | `260829/ms/models_ql/e5_s128/esmc_small_fold3-best.pt` | **0.12274** | 27 | **released** |

hwjang's decision was to re-select within the arm's own paper runs. Two facts make that a
re-selection rather than the import of an outside model, and both are checkable:

- **`260829` was already inside the candidate set.** The paper's ms arm pools two roots:
  `analysis/scoring/2_ms/per_run_results.csv` draws 9 of the arm's 15 `esmc_small` runs from
  `250527/1_ms_re` and 6 from `260829/ms` — seed 100 fold 4, and seed 128 folds 0-4. The winning
  run is one of the fifteen the published mean already averages over.
- **The criterion did not change, only the parse did.** Lowest validation loss, as it always was.

What is known about the two experiment roots, measured rather than assumed:

- They are **one experiment split by seed, not two experiments.** `250527/1_ms_re/models_ql/e5_s128`
  holds 0 of 5 `esmc_small` checkpoints — that seed never ran there — and `260829/run_bulk_ms.sh`
  says so in its header ("the e5_s128 runs that never happened … trained from scratch here …
  Nothing here writes into 250527/1_ms_re"). So *every* s128 run being in the newer root is the
  design, not a pattern. It also means root and seed are perfectly confounded: no amount of
  re-reading the logs can separate "seed 128 is better" from "the 2026 environment is different".
- **The environment was measured, and on this arm the drift is zero.** `260829` retrained three of
  `250527`'s s42 esmc_small cells from scratch (`drift_models/e5_s42/esmc_small_fold{0,1,2}`), 15
  months later. Every logged metric at the argmin epoch is identical to 5 decimal places — Train
  Acc, Val Acc, Train Loss, Val Loss and even the corrupt ROC-AUC — for all three folds. Only
  `Time:` differs (77-83 s against 137-145 s).
- **Same data.** Both roots' `config_ql/config_global.json` (md5 `5bbe0838a39f82c670091f0dcaa9167c`)
  and `config_esmc_small.py` (md5 `82933f8fe48355de2de7968571a693da`) are byte-identical and name
  the same absolute `ms_ql/train.csv`, whose mtime is 2025-05-25, before either run, 77,954 rows.
  The loaders agree at run time, not just in the config: `250527/1_ms_re` logs
  `Total samples: 77954` in 39 slurm files, and the 260829 ms job logs it in **18 of 18**, with no
  other value in that job.

  Look in the right job array. `260829` ran three jobs from one directory, and the ms one is
  **262623**; `slurm-262621_*` is the LOMO job and logs 155k-158k samples over per-allele splits.
  Grepping `260829/slurm-262621_*.out` for `77954` returns nothing and looks like a refutation of
  the paragraph above. It is the wrong array. `260829/run_bulk_ms.sh` is the script that says which
  is which, and its header states the design outright: the `e5_s128` runs "never happened" in
  `250527`, they are "trained from scratch here rather than resumed", and "Nothing here writes into
  250527/1_ms_re".

The decision was carried out by editing `ARMS` in `convert_checkpoints.py` and running
`--write --arm ms`, **never by renaming a staged file** — for the reason in step 3, a staged file
carries its staging name inside it. `models/ms_candidates/` still holds the four files that made the
comparison; they are not release files and are uploaded nowhere.

```bash
PY=/home/hwjang/miniforge3/envs/venv/bin/python
$PY tools/release/convert_checkpoints.py --list                    # sources exist, sizes right
$PY tools/release/convert_checkpoints.py --list --inspect-sources  # + structure, reads 2.5 GB
```

### 2. Produce the files

Before: the venv interpreter (torch 2.6.0). `.pixi/envs/default` has no torch. No GPU, no Slurm —
this is CPU-only I/O. About 20 seconds and 1.34 GB of disk for the eight release files, another
639 MB for the four staged ms candidates.

```bash
$PY tools/release/convert_checkpoints.py --write      # 8 files into models/
$PY tools/release/convert_checkpoints.py --stage-ms   # 4 files into models/ms_candidates/
```

One note about this machine rather than about the script: the login node's per-user memory cgroup
is tight and shared, and a single process that converts all four arms in a row was OOM-killed there
on 2026-09-10 at about 6.4 GB RSS. `--arm` one at a time is the reliable way, and it is what the
recorded runs below used:

```bash
for a in qualitative ms ic50_500 ic50_1000; do
  $PY tools/release/convert_checkpoints.py --check --arm $a --tmp-dir /home/hwjang/tmp_release
done
```

Both refuse to overwrite an existing output and exit 2, listing what is in the way. `--force`
replaces them and says how many. `--arm` and `--tier` narrow the job.

### 3. Prove the files are what they claim to be

Before: step 2, or files already in `models/` from an earlier run.

```bash
$PY tools/release/convert_checkpoints.py --check          # every file == a fresh derivation
$PY tools/release/convert_checkpoints.py --check-legacy   # this code produced the old files
$PY tools/release/convert_checkpoints.py --verify         # every file loads into the released model
```

Three different questions, and the ms arm is the reason all three exist.

**`--check`** re-derives each output from its source into a scratch directory and byte-compares,
then compares against the size/md5/sha256 recorded in `ARMS`. It writes nothing into `models/`. When
a demo-tier file differs it also reports the tensor-level diff, so "the container changed" is
distinguishable from "the weights changed". All eight files pass.

**`--verify`** answers a question `--check` cannot: is the file a *model*? It builds
`plm_cat_mean_inf` from the arm's own `configs/predict/*.py`, loads the file with
`load_state_dict(strict=True)`, and checks that the fp16 tier is bit-exactly `fp32.half()`, that the
fp32 tier is bit-exactly the source's `model_state_dict`, and that the source's stored `epoch` is
the corrected argmin epoch. Recorded 2026-09-10, all four arms: 64 tensors, 55,820,161 parameters,
strict load accepted, fp16 identical to `fp32.half()` on all 64 tensors as int16 bit patterns, no
NaN or Inf, round-trip max abs difference 4.88e-04. It also checks each predict config points at its
own arm's file, which is how the stale `config_ms.py` above is caught.

**Why the scratch copy keeps the same basename.** `torch.save` writes the output filename into the
file: the zip archive prefix is the basename stem, repeated once per record. Size and md5 therefore
depend on what a checkpoint was called when it was written. Two consequences:

- Renaming a release file with `mv` is wrong. It leaves a stale internal prefix and a hash nobody
  can reproduce. Re-emit from the source under the final name — which is what `--write` does.
- The four `models/prepi_esmc_small_*_fp16.pt` have three different sizes purely because their names
  have three different lengths. `--check` demonstrates this: against the new
  `prepibind_ic50_500_s128_f2_fp16.pt` (111,662,390 B) the old
  `prepi_esmc_small_ic50_500_e5_s128_f2_fp16.pt` (111,663,070 B) is byte-different and
  tensor-identical, key for key.

  The difference is accountable, with one wrinkle worth stating because it broke a first attempt at
  this check. An archive here has 68 records and writes the basename stem once into each, so a
  character of name normally costs 68 B: for the ic50_500 pair 68 x (31 - 41) = **-680 B**, observed
  -680 B; for the qualitative pair 68 x (34 - 32) = **+136 B**, observed +136 B. But `torch.save`
  also aligns each record's payload to 64 B, so a longer name occasionally pushes a record over a
  boundary and costs 68 + 64. The **ms** pair does exactly that: 68 x 10 + 64 = **744 B**, observed
  744 B. A scan of one source saved under basenames of length 20 to 39 gives +68 at every step
  except two, which give +132, and reproduces the exact size of three files in `models/`. So the
  rule is `68 x characters + 64 x k`, and `--check-legacy` reports `k` rather than assuming it is 0.

**So a new fp16 file is not byte-identical to the old one, and cannot be — that is the design, not
a fault.** No amount of care makes `prepibind_qualitative_s100_f0_fp16.pt` equal
`prepi_esmc_small_e5_s100_f0_fp16.pt` byte for byte; the names are inside the files. Asking for that
identity is asking for the wrong thing, and getting it would mean the rename had not happened.

The proof that the conversion path is unchanged runs the other way, and it is exact — this is
**`--check-legacy`**: re-emit each *old* name from the source that produced it, with today's
`demo/build_demo_assets.py checkpoint`, and you get the file that is in `models/` now, byte for
byte. Recorded 2026-09-10: **4 of 4**, including the superseded ms file from its own 250527 source.
Anything that had drifted in the conversion — torch version, dtype, key order, the dropping of the
optimizer state — would break it.

The ms arm has no such counterpart, because nothing was ever converted from
`260829/ms/.../fold3-best.pt` before. What stands in for it, all recorded above: the fp32 tier is
bit-identical to the source `model_state_dict`; the fp16 tier is bit-identical to `fp32.half()`;
both load `strict=True` into the model the predict config builds, 64 tensors and 55,820,161
parameters; the source's stored epoch is **27**, which is the corrected val_loss argmin epoch for
that run in `analysis/val_metrics.csv`; and the two files hash to

| file | bytes | md5 | sha256 |
|---|---:|---|---|
| `prepibind_ms_s128_f3.pt` | 223,301,834 | `fd30ab20b3e096f9ee0cca8ff8a0a1a1` | `b1f30d1f29e9a1077b39ab98f98c6458d1818caa0dffb8fecba5a461bbb06bb8` |
| `prepibind_ms_s128_f3_fp16.pt` | 111,661,918 | `5c72435dfb42b3ec0af6c75c8eb83ef4` | `bbc688ddc1a137c2ce6c0055ee4cbb25509efd16d22df213b22b515283e93fd2` |

Recorded so that the next person can re-check them, not because anything has been published.

### 4. Confirm the repository side is already renamed

Before uploading. The upload publishes filenames that the repository has to already name:
`configs/predict/*.py`, `README.md`, `demo/run.ipynb`, `demo/run_colab.ipynb` and
`THIRD_PARTY_NOTICES.md` must refer to `daylight-00/prepibind`, `daylight-00/prepibind-demo`,
`daylight-00/prepibind-embeddings` and the `prepibind_<arm>_<seed>_f<fold>[_fp16].pt` names.

```bash
grep -rn "prepi_esmc_small\|prepibind-esmc-300m" --include='*.py' --include='*.md' --include='*.ipynb' . | grep -v tools/release
```

should come back empty (`tools/release/` mentions the old names on purpose).

### The four superseded files in `models/`: removable, and not yet removed

**None of these has been deleted.** Nothing in the repository reads them any more —
`configs/predict/*.py` name the `prepibind_*` files — and every one is re-derivable in seconds from
its training checkpoint. They are listed here as removable so that whoever cleans up knows exactly
what and how much, and they are still on disk because `--check-legacy` is the evidence that the
conversion path is unchanged and it needs them present. Delete them after the upload is verified,
not before.

| file | bytes | MiB | relation to the release |
|---|---:|---:|---|
| `models/prepi_esmc_small_e5_s100_f0_fp16.pt` | 111,662,458 | 106.5 | **renamed**: same 64 tensors as `prepibind_qualitative_s100_f0_fp16.pt` |
| `models/prepi_esmc_small_ms_e5_s100_f1_fp16.pt` | 111,662,662 | 106.5 | **superseded, different weights** — see below |
| `models/prepi_esmc_small_ic50_500_e5_s128_f2_fp16.pt` | 111,663,070 | 106.5 | **renamed**: same 64 tensors as `prepibind_ic50_500_s128_f2_fp16.pt` |
| `models/prepi_esmc_small_ic50_1000_e5_s42_f1_fp16.pt` | 111,663,070 | 106.5 | **renamed**: same 64 tensors as `prepibind_ic50_1000_s42_f1_fp16.pt` |
| **total** | **446,651,260** | **426.0** | |

**`prepi_esmc_small_ms_e5_s100_f1_fp16.pt` is the one that is not a rename, and the distinction
matters more than the disk space.** It was built from
`250527/1_ms_re/models_ql/e5_s100/esmc_small_fold1-best.pt`, true val_loss 0.13128, and the ms arm
has since re-selected to `260829/ms/models_ql/e5_s128/esmc_small_fold3-best.pt`, 0.12274. Its
weights therefore differ from `prepibind_ms_s128_f3_fp16.pt` in **all 64 tensors**, not in a
filename. `--check` asserts that difference: for this arm, identical tensors would be the failure.
It must not be uploaded and must not be treated as an old name for the released ms model.

| | |
|---|---|
| file | `models/prepi_esmc_small_ms_e5_s100_f1_fp16.pt` |
| bytes | 111,662,662 (106.5 MiB) |
| md5 | `c87d0bad82c1fb1b75e2ca70bc5f61bb` |
| sha256 | `4f9248eb059929f4bf60ee0e3531926b8f3a7a0718cdb102b673c18575cd6856` |
| status | superseded 2026-09-10, kept on disk, publish nowhere |

`models/ms_candidates/` (4 files, 669,932,512 B, 638.9 MiB) is removable on the same terms: it holds
both sides of the comparison that produced the decision, under names that are not release names.
Keeping it until the upload is verified costs 639 MB and preserves the ability to re-run the
comparison.

`models/` is git-ignored, so none of them is tracked and removing them is not a repository change.
Confirmed on 2026-09-10 with `git check-ignore -v` on each file (all match `.gitignore:15:models`)
and `git ls-files models/`, which returns nothing.
The remaining entry in `models/`, `esmc_300m_2024_12_v0_fp16.pth`, is a **symlink** to
`/home/hwjang/project/PREpiBind/demo/models/`; it is the backbone, not a superseded file, and stays.

### 5. Upload

`upload_plan.md`, in order. Create and verify all three new repositories **before** deleting
anything. The two destructive steps are at the end and are marked; the second one destroys a
repository and its history irreversibly, though — corrected 2026-09-10, an earlier draft had this
wrong — every one of the four files it holds has a byte-identical local copy.

The eight files and the three cards are ready. Nothing else in this directory is a precondition.

Nobody in this workstream is authorised to run those commands. They need a human with write access
to `daylight-00`, and step 6 additionally needs the webserver owner to confirm which repository
`PREpiBind-web` actually reads.

## An improvement this script wants, which lives in someone else's file

`convert_checkpoints.py` does not reimplement the conversion. It imports
`demo/build_demo_assets.py` and calls `cmd_checkpoint`, so the float16 tier comes out of exactly the
code that produced every shipped fp16 checkpoint so far. `cmd_checkpoint` hard-codes `.half()`,
so the float32 tier currently uses a local fallback (`_emit_float32`) that repeats those four lines
with `.float()`. The script prints a note when it does.

The fallback disappears the moment `build_demo_assets.py` grows a dtype switch:

```python
c.add_argument("--dtype", choices=("float16", "float32"), default="float16")
...
cast = (lambda t: t.half()) if a.dtype == "float16" else (lambda t: t.float())
out  = {k: cast(v) if v.is_floating_point() else v for k, v in sd.items()}
```

`default="float16"` keeps every existing invocation byte-identical, and `convert_checkpoints.py`
detects the switch and routes both tiers through it with no change on this side.
`demo/build_demo_assets.py` is not this directory's file, so the change is proposed, not made.

## Things that are true and easy to get wrong

- The store published as `daylight-00/prepibind-embeddings` is **not** bit-identical to what the
  released checkpoints were trained on. Three of its 154 keys (`H2-IAdA`, `H2-IAdB`, `H2-IAg7A`)
  were corrected on 2026-09-08 and the file carries no attribute recording it. Disclosed in all
  three cards; governed by `IMG/docs/decisions/h2-fix-is-not-retro-applied.md`.
- `daylight-00/esmc-300m-2024-12` holds a bf16 backbone with **no copy on local disk**. Edit that
  repository in place; never delete and recreate it.
- The two `emb_hla_esmc_small_*0601*.h5` are webserver assets whose content is already published as
  `daylight-00/emb_hla_esmc_small_0601_fp16` (7,282 `.npy` shards). Removing them from the backbone
  repository is a deletion, not a migration.
- `huggingface-cli` no longer runs. The CLI is `hf` (huggingface_hub 1.18.0), in the venv.
