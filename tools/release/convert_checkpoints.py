#!/usr/bin/env python3
"""The four training checkpoints -> the eight files that get published.

    python tools/release/convert_checkpoints.py --write         # produce them
    python tools/release/convert_checkpoints.py --check         # prove the ones on disk are those
    python tools/release/convert_checkpoints.py --check-legacy  # prove this is the code that made
                                                                # the old prepi_esmc_small_* files
    python tools/release/convert_checkpoints.py --manifest      # the table upload_plan.md quotes
    python tools/release/convert_checkpoints.py --stage-ms      # both ms candidates, side by side

Two tiers out of one source per arm:

    research  float32, optimizer state dropped   prepibind_<arm>_<seed>_f<fold>.pt       -> daylight-00/prepibind
    demo      float16, optimizer state dropped   prepibind_<arm>_<seed>_f<fold>_fp16.pt  -> daylight-00/prepibind-demo

Nothing here is a new measurement. Both tiers are pure functions of the training checkpoint named
in ARMS, so `--check` can prove it: it re-derives each file into a scratch directory *under the
same basename* and byte-compares. Read on for why the basename matters.

Three things this script exists to get right
--------------------------------------------

1. **The conversion itself is not reimplemented here.** `demo/build_demo_assets.py checkpoint` is
   the code that produced every fp16 checkpoint shipped so far; this script imports that module and
   calls `cmd_checkpoint`, so the demo tier comes out of exactly the same four lines as before. Only
   the float32 tier needs anything new, and it needs one word: `.float()` instead of `.half()`. If
   `build_demo_assets.py` grows the `--dtype` switch, this script detects it and routes both tiers
   through it, and the local fallback stops being used. See `_emit`.

2. **`torch.save` embeds the output filename in the file.** The zip archive prefix is the basename
   stem, repeated once per record, so both the size and the md5 of a checkpoint depend on what it
   was called when it was written. Renaming a released file with `mv` therefore leaves a file whose
   internal prefix disagrees with its name and whose published hash is no longer reproducible.
   Every file here is re-emitted from its source under its final name, and `--check` re-derives
   into `<scratch>/<same basename>` for the same reason. It is also why the four current
   `models/prepi_esmc_small_*.pt` have three different sizes: filename length, not content. The
   basename appears once per zip record, 68 of them (64 tensors + `data.pkl`, `version`,
   `byteorder`, `.format_version`), so one more character costs 68 B, plus an occasional 64 B when
   the longer name pushes a record past a payload-alignment boundary (measured: at basename lengths
   22 and 26 of a 20..39 scan). `--check-legacy` decomposes each pair's size difference that way
   and complains if anything is left over.

   The consequence worth stating plainly, because it is the natural thing to ask for and it cannot
   be delivered: **a release file can never be byte-identical to the legacy file it replaces.**
   `prepibind_qualitative_s100_f0_fp16.pt` and `prepi_esmc_small_e5_s100_f0_fp16.pt` hold the same
   64 tensors, byte for byte after decoding, and differ by 136 B, which is 68 x the 2-character
   difference in their names. The ms pair differs by 744 B = 68 x 10 + 64, one alignment step, and
   by every one of its 64 tensors, because that one is a re-selection and not a rename. Proving
   the conversion path therefore means re-deriving the *legacy* file under the *legacy* name, which
   is `--check-legacy`, and comparing the new file to the old one at the tensor level, which
   `--check` does.

3. **It refuses to overwrite.** `--write` stops if any output already exists; `--force` is the only
   way past, and it says which files it replaced. Re-emitting is cheap, but silently replacing a
   file that a published hash refers to is not.

Selection rule for the four sources: lowest validation loss within the arm, across the 15 runs
(3 seeds x 5 folds). Not test-set anything, and never `val_auc`, which is corrupt in
`analysis/scoring/*/per_run_results.csv` and in the training logs alike (the `roc_auc_score`
arguments are swapped). The seed is the one in the run directory name `e5_s<seed>`, not the one in
`config_global.json`: `run.sh` passes `--seed`/`--lr` on the command line and the recorded config
still says the defaults it overrode.

The `val_loss` values in `ARMS` are read from `analysis/val_metrics.csv`, regenerated 2026-09-10
after a regex in `analysis/tools/parse_train_logs.py` was fixed. That regex required `[\\d.]+` for
`Val ROC-AUC`, so every epoch line whose (corrupt, negative) ROC-AUC carried a minus sign was
dropped, and the recorded per-run minimum was a minimum over a filtered subset: too high, on 46.7 %
of all runs. Three of the four checkpointed epochs already agreed with the corrected argmins and
not with the old ones, which is the independent check that the corrected numbers are the right
ones. Nothing about the four source checkpoints changed -- only the numbers printed beside them.

**ms was re-selected on 2026-09-10 and is now written like the others.** hwjang decided to
re-select within the arm's own paper runs, which resolves to `260829/ms/models_ql/e5_s128/
esmc_small_fold3-best.pt` (val_loss 0.12274, epoch 27) in place of `250527/1_ms_re/models_ql/
e5_s100/esmc_small_fold1-best.pt` (0.13128, epoch 20). Two facts settled it and both are checkable
here rather than taken on trust:

* **260829 was already inside the candidate set, not outside it.** The paper's ms arm pools two
  roots: `analysis/scoring/2_ms/per_run_results.csv` draws 9 runs from `250527/1_ms_re` and 6 from
  `260829/ms` (s100 f4 and s128 f0-f4). Re-selecting does not import a run from somewhere new.
* **The two roots are comparable.** `config_global.json` and `config_esmc_small.py` are
  byte-identical across them, both log `Total samples: 77954` from the one `ms_ql/train.csv`, and
  260829's `drift_models` replicate of 250527's s42 folds 0-2 reproduces them exactly -- the same
  three val_loss values, 0.14181 / 0.14277 / 0.14074, in `analysis/val_metrics.csv`.

The `disputed` flag stays in `Arm` and is now unused. It is the mechanism for the next arm whose
source is questioned; leaving it costs nothing and re-deriving it under pressure costs a mistake.

**The old ms file is superseded, not renamed.** `models/prepi_esmc_small_ms_e5_s100_f1_fp16.pt`
holds different weights, from the 250527 run. It is kept on disk until hwjang commits the release
and it must never be uploaded. `--check` asserts that its tensors *differ* from the new ms file;
see `LEGACY`.

Environment: the venv interpreter, torch 2.6.0.

    /home/hwjang/miniforge3/envs/venv/bin/python tools/release/convert_checkpoints.py --check

`.pixi/envs/default` has no torch. No GPU and no Slurm job is involved: this is CPU-only I/O.
"""
import argparse
import filecmp
import hashlib
import importlib.util
import inspect
import os
import shutil
import sys
import tempfile
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
BUILDER = os.path.join(REPO, "demo", "build_demo_assets.py")

#: Where the outputs go. `models/` is already ignored by `.gitignore` (the bare `models` rule
#: matches at any depth), and it is where `configs/predict/*.py` look for the fp16 checkpoints, so
#: the demo tier lands ready to run. No new ignore rule is needed and none was added.
DEFAULT_OUT = os.path.join(REPO, "models")

#: The size every training checkpoint has: model_state_dict + optimizer_state_dict + epoch.
SRC_BYTES = 669_924_074
#: 64 float32 tensors, this many elements. Same head in all four arms.
N_PARAMS = 55_820_161
#: `torch.save` writes the output basename once per zip record: 64 tensors + `data.pkl` + `version`
#: + `byteorder` + `.format_version` = 68. So one more character of basename costs 68 B --
#: *usually*. `torch.save` also aligns each record's payload to a 64 B boundary, so a longer name
#: occasionally pushes a record across a boundary and costs 68 + 64 instead.
#:
#: Measured on this machine, one source, basenames of length 20..39 (torch 2.6.0, fp16 tier):
#: +68 at every length except 22 and 26, which cost +132. The scan reproduces the exact size of
#: three files in `models/` -- 111,661,918 at 25 characters, 111,662,390 at 31, 111,662,662 at 35.
#: So `--check-legacy` decomposes an observed size difference as 68 x characters + 64 x k, and only
#: complains if the remainder is not a whole number of alignment steps -- that would mean the two
#: files differ by something other than their names.
ZIP_BYTES_PER_CHAR = 68
ZIP_ALIGN = 64


class Arm:
    """One released model: where it came from, what it is called, what it hashes to."""

    def __init__(self, name, seed, fold, val_loss, epoch, src, research, demo, disputed=False):
        self.name, self.seed, self.fold = name, seed, fold
        self.val_loss, self.epoch, self.src = val_loss, epoch, src
        self.expect = {"research": research, "demo": demo}
        #: True while the arm's source checkpoint is not settled. A disputed arm is excluded from
        #: `--write`/`--check` entirely: emitting it would put a release name on an undecided file.
        self.disputed = disputed

    def stem(self):
        return f"prepibind_{self.name}_{self.seed}_f{self.fold}"

    def filename(self, tier):
        return self.stem() + (".pt" if tier == "research" else "_fp16.pt")


#: (bytes, md5, sha256) recorded when each artifact was first produced under its final name, with
#: torch 2.6.0. `--check` reports a mismatch against these as well as against a fresh derivation:
#: the fresh derivation is the real test, these catch "same content, different producer".
def _e(nbytes, md5, sha256):
    return SimpleNamespace(bytes=nbytes, md5=md5, sha256=sha256)


ARMS = [
    Arm("qualitative", "s100", 0, 0.35582, 20,
        "/home/hwjang/project/250513/1_bulk/2/models/e5_s100/esmc_small_fold0-best.pt",
        _e(223_302_574, "e0d105b190e03dcc54d889735bd65099",
           "53f47f56ad9bb8632921bce1d094fbf404abf9d8aec0d0a4862c4cc4ff8d0574"),
        _e(111_662_594, "814b2c2be146d77d558c96e823a8fcfe",
           "aaf849fc0c4dc1f92426b4681fb08ea5d2c01467870d90ece824e8870410e3db")),
    #: Re-selected 2026-09-10 (hwjang). Was s100 fold 1 from 250527, val_loss 0.13128 -- the
    #: filtered-minimum parser had hidden this run's true 0.12274. See MS_DECISION below.
    Arm("ms", "s128", 3, 0.12274, 27,
        "/home/hwjang/project/260829/ms/models_ql/e5_s128/esmc_small_fold3-best.pt",
        _e(223_301_834, "fd30ab20b3e096f9ee0cca8ff8a0a1a1",
           "b1f30d1f29e9a1077b39ab98f98c6458d1818caa0dffb8fecba5a461bbb06bb8"),
        _e(111_661_918, "5c72435dfb42b3ec0af6c75c8eb83ef4",
           "bbc688ddc1a137c2ce6c0055ee4cbb25509efd16d22df213b22b515283e93fd2")),
    Arm("ic50_500", "s128", 2, 0.46631, 28,
        "/home/hwjang/project/250516/7_ic50_etc/models_500/e5_s128/esmc_small_fold2-best.pt",
        _e(223_302_370, "c589a329ceaf3120752c5c78e3062cb8",
           "f43be97abf25f1aead37c50620a1111e725c9650fafcb2f3241724b91ff539ad"),
        _e(111_662_390, "3523cff0cd48e1fbe8ef4afd943d7860",
           "be2aa51fa41c88cc1a86624cfd51a497af867c75877ad1a1878ad7bf02deb14e")),
    Arm("ic50_1000", "s42", 1, 0.49456, 28,
        "/home/hwjang/project/250516/7_ic50_etc/models_1000/e5_s42/esmc_small_fold1-best.pt",
        _e(223_302_370, "708a3753ffae86beec884b31e62401fc",
           "e48e60f2fc11e941cae703c28af7937e0e4760422554ddd39fdc9ed69a3451fb"),
        _e(111_662_390, "4809f227098e5fb0f6417e951a4cc1bd",
           "0305b41261250523de7d955f6d75baabd952081f742088905227e0dad785aa4f")),
]

#: What each arm's fp16 file used to be called, and -- the part that carries information -- whether
#: the old file is *the same model under the old name* or a superseded one.
#:
#: `same_tensors=True` (three arms): the release file and the legacy file hold identical tensors and
#: differ only in bytes, because the basename is written into the zip archive 68 times (gotcha 2).
#: `--check` asserts the tensors are equal; if they ever stopped being equal, the rename would have
#: changed the model.
#:
#: `same_tensors=False` (ms only): the legacy file was built from a *different* training checkpoint
#: and is superseded, not renamed. `--check` asserts the tensors DIFFER -- equality there would mean
#: the re-selection silently did not happen.
#:
#: `src` is the training checkpoint each legacy file was built from. `--check-legacy` re-emits from
#: it under the legacy basename and byte-compares against the legacy file itself, which is the only
#: available proof that the code here is the code that produced the shipped files. It cannot be done
#: by comparing the new name against the old one: those necessarily differ by 68 B per character of
#: basename, whatever the tensors are.
LEGACY = {
    "qualitative": SimpleNamespace(
        file="prepi_esmc_small_e5_s100_f0_fp16.pt", same_tensors=True,
        src="/home/hwjang/project/250513/1_bulk/2/models/e5_s100/esmc_small_fold0-best.pt",
        reason="renamed only: prepibind_qualitative_s100_f0_fp16.pt is the same tensors"),
    "ms": SimpleNamespace(
        file="prepi_esmc_small_ms_e5_s100_f1_fp16.pt", same_tensors=False,
        src="/home/hwjang/project/250527/1_ms_re/models_ql/e5_s100/esmc_small_fold1-best.pt",
        reason="SUPERSEDED 2026-09-10, not renamed: built from 250527 s100 fold 1 "
               "(true val_loss 0.13128). The ms arm re-selected to 260829 s128 fold 3 "
               "(0.12274) once the parser stopped dropping negative-ROC-AUC epoch lines. "
               "Different weights, not a different name. Do not publish; do not delete "
               "until hwjang has committed the release."),
    "ic50_500": SimpleNamespace(
        file="prepi_esmc_small_ic50_500_e5_s128_f2_fp16.pt", same_tensors=True,
        src="/home/hwjang/project/250516/7_ic50_etc/models_500/e5_s128/esmc_small_fold2-best.pt",
        reason="renamed only: prepibind_ic50_500_s128_f2_fp16.pt is the same tensors"),
    "ic50_1000": SimpleNamespace(
        file="prepi_esmc_small_ic50_1000_e5_s42_f1_fp16.pt", same_tensors=True,
        src="/home/hwjang/project/250516/7_ic50_etc/models_1000/e5_s42/esmc_small_fold1-best.pt",
        reason="renamed only: prepibind_ic50_1000_s42_f1_fp16.pt is the same tensors"),
}

#: The predict config that names each arm's demo checkpoint. `--verify` builds the model from the
#: config's `model_args` and loads each release file into it with `strict=True`, and it checks that
#: the config's `chkp_path` is in fact this arm's demo file. That second check is why the map exists:
#: a re-selected arm whose config still names the old file is a release that silently runs the wrong
#: weights, and nothing else in this repository would notice.
PREDICT_CONFIG = {
    "qualitative": "configs/predict/config_demo.py",
    "ms": "configs/predict/config_ms.py",
    "ic50_500": "configs/predict/config_ic50_500.py",
    "ic50_1000": "configs/predict/config_ic50_1000.py",
}

TIERS = ("research", "demo")
DTYPE = {"research": "float32", "demo": "float16"}

#: Where `--stage-ms` puts the two ms candidates, under names that cannot be mistaken for a release
#: name and cannot be renamed into one (gotcha 2: the stem is inside the file, so picking a winner
#: means re-emitting it, not `mv`). The choice was made on 2026-09-10 and `ARMS` now names the
#: winner; this staging area is kept because it is what the comparison was made from, and
#: re-running `--stage-ms` reproduces both sides of it.
MS_STAGING = os.path.join(DEFAULT_OUT, "ms_candidates")

MS_CANDIDATES = [
    SimpleNamespace(
        key="A_released", seed="s100", fold=1, val_loss=0.13128, epoch=20,
        stem="ms_candidate_A_released_250527_s100_f1",
        src="/home/hwjang/project/250527/1_ms_re/models_ql/e5_s100/esmc_small_fold1-best.pt",
        note="NOT SELECTED (2026-09-10). The previously released ms model, and the source of the "
             "superseded models/prepi_esmc_small_ms_e5_s100_f1_fp16.pt"),
    SimpleNamespace(
        key="B_argmin", seed="s128", fold=3, val_loss=0.12274, epoch=27,
        stem="ms_candidate_B_argmin_260829_s128_f3",
        src="/home/hwjang/project/260829/ms/models_ql/e5_s128/esmc_small_fold3-best.pt",
        note="SELECTED (2026-09-10). The true val_loss argmin of the ms arm once the parser regex "
             "is fixed; released as prepibind_ms_s128_f3[_fp16].pt"),
]


# --------------------------------------------------------------------------- the builder module

def load_builder():
    """`demo/build_demo_assets.py`, imported by path. It is a script, not a package member."""
    if not os.path.exists(BUILDER):
        sys.exit(f"cannot find the conversion code: {BUILDER}")
    spec = importlib.util.spec_from_file_location("build_demo_assets", BUILDER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def builder_supports_dtype(bda):
    """Has `cmd_checkpoint` grown the float32 switch yet?

    Checked by reading the source rather than by calling it: an unpatched `cmd_checkpoint` ignores
    an unknown attribute on its namespace and returns float16 regardless, which would silently
    publish half-precision weights as the research tier.
    """
    try:
        return "dtype" in inspect.getsource(bda.cmd_checkpoint)
    except (OSError, TypeError):
        return False


def _emit(bda, src, dst, tier):
    """Write one release file. The demo tier is `build_demo_assets.py`, unchanged."""
    if builder_supports_dtype(bda):
        return bda.cmd_checkpoint(SimpleNamespace(src=src, dst=dst, dtype=DTYPE[tier]))
    if tier == "demo":
        return bda.cmd_checkpoint(SimpleNamespace(src=src, dst=dst))
    return _emit_float32(bda, src, dst)


def _emit_float32(bda, src, dst):
    """`cmd_checkpoint` with `.float()` where it has `.half()`.

    Only reachable while `demo/build_demo_assets.py` has no `--dtype`. The three lines are copied
    deliberately and are meant to be deleted: adding

        c.add_argument("--dtype", choices=("float16", "float32"), default="float16")
        cast = (lambda t: t.half()) if a.dtype == "float16" else (lambda t: t.float())

    to `cmd_checkpoint` makes `_emit` route here instead, and `default="float16"` keeps every
    existing invocation byte-identical. Sources are float32 already, so this tier is a pure strip:
    optimizer state and epoch dropped, values untouched.
    """
    import torch
    ck = torch.load(src, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]
    out = {k: v.float() if v.is_floating_point() else v for k, v in sd.items()}
    n = sum(v.numel() for v in out.values())
    torch.save({"model_state_dict": out}, dst)
    print(f"  {n / 1e6:.2f} M parameters, epoch {ck.get('epoch')}")
    print(f"  {os.path.getsize(src) / 1048576:.1f} MB -> {os.path.getsize(dst) / 1048576:.1f} MB"
          f"  (optimizer state dropped, weights cast to float32)")
    print(f"  -> {dst}  md5 {bda.md5(dst)}")
    return 0


# --------------------------------------------------------------------------------------- hashing

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------------------------ the checks

def check_sources(arms):
    """Every source readable, the right size, and the structure all four share. No torch load."""
    bad = 0
    for a in arms:
        if not os.path.exists(a.src):
            print(f"  MISSING  {a.name}: {a.src}")
            bad += 1
            continue
        n = os.path.getsize(a.src)
        flag = "" if n == SRC_BYTES else f"  <- expected {SRC_BYTES:,}"
        if flag:
            bad += 1
        if not os.access(a.src, os.R_OK):
            print(f"  UNREADABLE  {a.name}: {a.src}")
            bad += 1
        print(f"  {a.name:<12} {n:>12,} B  {a.src}{flag}")
    return bad


def inspect_source(src):
    """(n_tensors, dtypes, numel, epoch) of a training checkpoint."""
    import torch
    ck = torch.load(src, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]
    return (len(sd), sorted({str(v.dtype) for v in sd.values()}),
            sum(v.numel() for v in sd.values()), ck.get("epoch"))


def tensor_diff(a_path, b_path):
    """Key-by-key comparison of two release files. Returns a list of human-readable differences."""
    import torch
    x = torch.load(a_path, map_location="cpu", weights_only=False)["model_state_dict"]
    y = torch.load(b_path, map_location="cpu", weights_only=False)["model_state_dict"]
    out = []
    if list(x) != list(y):
        missing, extra = set(x) - set(y), set(y) - set(x)
        if missing or extra:
            out.append(f"key sets differ ({len(missing)} missing, {len(extra)} extra)")
        else:
            out.append("key order differs")
    for k in [k for k in x if k in y]:
        if x[k].dtype != y[k].dtype:
            out.append(f"{k}: dtype {x[k].dtype} vs {y[k].dtype}")
        elif not torch.equal(x[k], y[k]):
            out.append(f"{k}: values differ")
    return out


def check_one(bda, arm, tier, out_dir, tmp_root):
    """Re-derive one artifact and byte-compare. Writes only inside `tmp_root`. Returns 0 if ok."""
    name = arm.filename(tier)
    dst = os.path.join(out_dir, name)
    exp = arm.expect[tier]
    print(f"[{tier:<8}] {name}")
    if not os.path.exists(dst):
        print(f"  MISSING  {dst}")
        return 1

    have_bytes = os.path.getsize(dst)
    have_md5, have_sha = bda.md5(dst), sha256(dst)
    print(f"  on disk   {have_bytes:>12,} B  md5 {have_md5}")

    bad = 0
    if have_bytes != exp.bytes or have_md5 != exp.md5 or have_sha != exp.sha256:
        print(f"  RECORDED  {exp.bytes:>12,} B  md5 {exp.md5}  <- differs from the recorded value")
        bad += 1

    scratch = os.path.join(tmp_root, tier, name)          # same basename: the stem is in the file
    os.makedirs(os.path.dirname(scratch), exist_ok=True)
    _emit(bda, arm.src, scratch, tier)
    if filecmp.cmp(dst, scratch, shallow=False):
        print("  byte-identical to a fresh derivation from the source checkpoint")
    else:
        print(f"  DIFFERS from a fresh derivation ({os.path.getsize(scratch):,} B, "
              f"md5 {bda.md5(scratch)})")
        for line in tensor_diff(dst, scratch) or ["tensors are equal: the difference is the "
                                                  "container (filename stem, or torch version)"]:
            print(f"    {line}")
        bad += 1

    lg = LEGACY[arm.name]
    legacy = os.path.join(out_dir, lg.file)
    if tier == "demo" and os.path.exists(legacy):
        d = tensor_diff(dst, legacy)
        note = "tensors identical" if not d else f"{len(d)} tensor difference(s): {d[0]}"
        print(f"  vs {lg.file}: {note} ({os.path.getsize(legacy):,} B)")
        if lg.same_tensors and d:
            print("    EXPECTED IDENTICAL: the legacy file is this model under its old name.")
            bad += 1
        elif lg.same_tensors:
            print(f"    ...and {os.path.getsize(legacy) - have_bytes:+,} B of container, from "
                  f"{len(lg.file) - len(name)} characters of basename. Two names cannot give one "
                  f"file; --check-legacy decomposes the difference and is the byte proof.")
        elif not d:
            print("    EXPECTED TO DIFFER: this arm was re-selected, so identical tensors would "
                  "mean the new source was not used.")
            bad += 1
        else:
            print(f"    superseded, as intended. {lg.reason}")
    os.remove(scratch)
    return bad


# ------------------------------------------------------------------------------------- the modes

def do_write(bda, arms, tiers, out_dir, force):
    os.makedirs(out_dir, exist_ok=True)
    planned = [(a, t) for a in arms for t in tiers]
    existing = [(a, t) for a, t in planned if os.path.exists(os.path.join(out_dir, a.filename(t)))]
    if existing and not force:
        print("refusing to overwrite:")
        for a, t in existing:
            print(f"  {os.path.join(out_dir, a.filename(t))}")
        print("\n--check verifies them against a fresh derivation without writing anything.")
        print("--force replaces them.")
        return 2
    if existing:
        print(f"--force: replacing {len(existing)} existing file(s)\n")

    for a, t in planned:
        dst = os.path.join(out_dir, a.filename(t))
        print(f"[{t:<8}] {a.name}  seed {a.seed} fold {a.fold}  val_loss {a.val_loss}")
        print(f"  src {a.src}")
        _emit(bda, a.src, dst, t)
        print(f"  sha256 {sha256(dst)}")
    return 0


def do_check(bda, arms, tiers, out_dir, tmp_dir):
    bad = 0
    tmp_root = tempfile.mkdtemp(prefix="prepibind_release_", dir=tmp_dir)
    print(f"scratch: {tmp_root}  (removed at the end; nothing is written to {out_dir})\n")
    try:
        for a in arms:
            for t in tiers:
                bad += check_one(bda, a, t, out_dir, tmp_root)
                print()
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
    print("ok" if not bad else f"{bad} problem(s)")
    return 1 if bad else 0


def load_predict_config(arm):
    """The arm's `configs/predict/*.py`, imported by path."""
    path = os.path.join(REPO, PREDICT_CONFIG[arm.name])
    spec = importlib.util.spec_from_file_location(f"cfg_{arm.name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return path, mod.config


def do_verify(bda, arms, tiers, out_dir):
    """Structural verification: does each file *load into the released model*, strict?

    `--check` proves a file is a faithful function of its source, and `--check-legacy` proves this
    code produced the files that already shipped. Neither is available for a source that has never
    been converted before -- which is exactly the ms arm after the 2026-09-10 re-selection. There
    is no old ms file to reproduce, so "byte-identical to what we shipped" is not a question that
    can be asked about it.

    What can be asked, and is asked here:

    * every key and shape matches the model that `configs/predict/*.py` builds, and
      `load_state_dict(strict=True)` accepts it -- no missing, unexpected or mis-shaped tensor;
    * the fp16 tier is *exactly* the fp32 tier cast down: `fp16[k] == fp32[k].half()` for all 64,
      compared as bit patterns, which fails on any NaN introduced by the cast;
    * the fp32 tier is exactly the source's `model_state_dict`, and the source's stored `epoch`
      equals the epoch of the val_loss argmin in `analysis/val_metrics.csv` (recorded in ARMS);
    * the predict config for the arm points at the arm's demo file, not at a superseded one.
    """
    import torch
    bad = 0
    for a in arms:
        print(f"[{a.name}] seed {a.seed} fold {a.fold}  val_loss {a.val_loss}  epoch {a.epoch}")
        cfg_path, cfg = load_predict_config(a)
        net = cfg["model"](**cfg["model_args"])
        ref = net.state_dict()
        print(f"  model {cfg['model'].__name__}({', '.join(f'{k}={v}' for k, v in cfg['model_args'].items())})"
              f"  from {os.path.relpath(cfg_path, REPO)}")
        print(f"  reference head: {len(ref)} tensors, {sum(v.numel() for v in ref.values()):,} parameters")

        want = os.path.basename(a.filename("demo"))
        got = os.path.basename(str(cfg["Test"]["chkp_path"]))
        if got == want:
            print(f"  config chkp_path -> {got}  [this arm's demo file]")
        else:
            print(f"  config chkp_path -> {got}  <- WRONG, expected {want}")
            print(f"     {cfg_path} is outside this script's scope; reported, not edited.")
            bad += 1

        src_sd = src_epoch = None
        loaded = {}
        for t in tiers:
            path = os.path.join(out_dir, a.filename(t))
            if not os.path.exists(path):
                print(f"  MISSING {path}")
                bad += 1
                continue
            obj = torch.load(path, map_location="cpu", weights_only=False)
            if set(obj) != {"model_state_dict"}:
                print(f"  [{t}] unexpected top-level keys: {sorted(obj)}  <- optimizer state or "
                      f"epoch should not survive conversion")
                bad += 1
            sd = obj["model_state_dict"]
            loaded[t] = sd
            dts = sorted({str(v.dtype) for v in sd.values()})
            n = sum(v.numel() for v in sd.values())
            ok = dts == [f"torch.{DTYPE[t]}"] and n == N_PARAMS and len(sd) == len(ref)
            print(f"  [{t:<8}] {len(sd)} tensors, {dts}, {n:,} parameters, "
                  f"{os.path.getsize(path):,} B  [{'ok' if ok else 'UNEXPECTED'}]")
            if not ok:
                bad += 1
            shape_bad = [k for k in ref if k not in sd or tuple(sd[k].shape) != tuple(ref[k].shape)]
            extra = [k for k in sd if k not in ref]
            if shape_bad or extra:
                print(f"    {len(shape_bad)} missing/mis-shaped, {len(extra)} unexpected: "
                      f"{(shape_bad + extra)[:3]}")
                bad += 1
            else:
                print(f"    all {len(ref)} keys and shapes match the reference head")
            try:
                cfg["model"](**cfg["model_args"]).load_state_dict(sd, strict=True)
                print("    load_state_dict(strict=True): accepted")
            except Exception as exc:                                  # noqa: BLE001
                print(f"    load_state_dict(strict=True): REJECTED  {exc}")
                bad += 1
            print(f"    md5 {bda.md5(path)}")
            print(f"    sha256 {sha256(path)}")

        if src_sd is None and os.path.exists(a.src):
            ck = torch.load(a.src, map_location="cpu", weights_only=False)
            src_sd, src_epoch = ck["model_state_dict"], ck.get("epoch")
            del ck
        if src_epoch is not None:
            mark = "matches the corrected val_loss argmin" if src_epoch == a.epoch \
                   else f"<- MISMATCH, ARMS says {a.epoch}"
            print(f"  source stored epoch {src_epoch}  [{mark}]")
            if src_epoch != a.epoch:
                bad += 1

        if "research" in loaded and src_sd is not None:
            d = [k for k in src_sd if not torch.equal(loaded["research"][k], src_sd[k].float())]
            print(f"  fp32 vs source model_state_dict: "
                  f"{'identical on all 64 tensors' if not d else f'{len(d)} differ: {d[:2]}'}")
            if d:
                bad += 1

        if "research" in loaded and "demo" in loaded:
            f32, f16 = loaded["research"], loaded["demo"]
            d = [k for k in f32 if not torch.equal(f16[k].view(torch.int16),
                                                   f32[k].half().view(torch.int16))]
            print(f"  fp16 vs fp32.half() bit patterns: "
                  f"{'identical on all 64 tensors' if not d else f'{len(d)} differ: {d[:2]}'}")
            if d:
                bad += 1
            worst = max((f32[k] - f16[k].float()).abs().max().item() for k in f32)
            rel = max(((f32[k] - f16[k].float()).abs().max()
                       / f32[k].abs().max().clamp(min=1e-12)).item() for k in f32)
            nan = sum(int(torch.isnan(f16[k]).any() or torch.isinf(f16[k]).any()) for k in f16)
            print(f"  fp32 -> fp16 round-trip: max abs {worst:.3e}, max rel {rel:.3e}, "
                  f"{nan} tensor(s) with NaN/Inf")
            if nan:
                bad += 1
        print()
    print("ok" if not bad else f"{bad} problem(s)")
    return 1 if bad else 0


def do_check_legacy(bda, arms, out_dir, tmp_dir):
    """Prove this code is the code that produced the fp16 files already in `models/`.

    The obvious check -- "is `prepibind_<arm>_..._fp16.pt` byte-identical to
    `prepi_esmc_small_..._fp16.pt`?" -- can never pass and proves nothing when it fails: the two
    names differ, the basename is inside the archive 68 times, so the files differ by
    68 B x (difference in basename length) no matter what the tensors are. It is a test of the
    filenames, not of the conversion.

    The test that does bite: re-emit from the same training checkpoint **under the legacy
    basename** and byte-compare with the legacy file. Passing means every byte the old file has,
    this code reproduces -- same source, same cast, same dtype, same key order, same torch. That is
    what "the conversion path is the one that produced them" means, and it is checkable for the
    three renamed arms.

    The ms arm cannot pass and must not: its legacy file was built from the 250527 checkpoint that
    is no longer the arm's selection. Re-emitting from *its own* source still reproduces it byte for
    byte, which is the useful thing -- it shows the path is sound there too, so the only difference
    between the old ms file and the new one is the deliberate change of source.
    """
    bad = 0
    tmp_root = tempfile.mkdtemp(prefix="prepibind_legacy_", dir=tmp_dir)
    print(f"scratch: {tmp_root}  (removed at the end; nothing is written to {out_dir})\n")
    try:
        for a in arms:
            lg = LEGACY[a.name]
            legacy = os.path.join(out_dir, lg.file)
            release = os.path.join(out_dir, a.filename("demo"))
            print(f"[{a.name}] {lg.file}")
            if not os.path.exists(legacy):
                print("  absent -- already removed; nothing to compare\n")
                continue

            n_legacy = os.path.getsize(legacy)
            print(f"  on disk   {n_legacy:>12,} B  md5 {bda.md5(legacy)}")
            print(f"  built from {lg.src}")
            if lg.src != a.src:
                print(f"  NOTE: the arm's source is now {a.src}")

            if not os.path.exists(lg.src):
                print("  SOURCE MISSING, cannot re-derive\n")
                bad += 1
                continue
            scratch = os.path.join(tmp_root, lg.file)     # legacy basename: that is the point
            _emit(bda, lg.src, scratch, "demo")
            if filecmp.cmp(legacy, scratch, shallow=False):
                print("  byte-identical to a fresh derivation under its own basename: this code "
                      "is what produced it")
            else:
                print(f"  DIFFERS from a fresh derivation ({os.path.getsize(scratch):,} B)")
                for line in tensor_diff(legacy, scratch) or ["tensors equal; container differs"]:
                    print(f"    {line}")
                bad += 1
            os.remove(scratch)

            if os.path.exists(release):
                n_rel = os.path.getsize(release)
                d_char = len(lg.file) - len(a.filename("demo"))
                delta = n_legacy - n_rel
                resid = delta - ZIP_BYTES_PER_CHAR * d_char
                steps, rem = divmod(resid, ZIP_ALIGN)
                accounted = rem == 0
                how = f"{ZIP_BYTES_PER_CHAR} x {d_char} characters of basename"
                if steps:
                    how += f" + {ZIP_ALIGN} x {steps} alignment step(s)"
                mark = "fully accounted for by the name" if accounted else \
                       f"UNEXPLAINED remainder {rem:+,} B -- not an alignment step"
                print(f"  vs {a.filename('demo')} ({n_rel:,} B): {delta:+,} B = {how}  [{mark}]")
                if not accounted:
                    bad += 1
                if lg.same_tensors:
                    print("  -> same tensors, different name, therefore different bytes. "
                          "Byte-identity between the two names is not achievable.")
                else:
                    print(f"  -> {lg.reason}")
            print()
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
    print("ok" if not bad else f"{bad} problem(s)")
    return 1 if bad else 0


def do_stage_ms(bda, tiers, out_dir, force):
    """Both ms candidates, side by side, under staging names. Publishes nothing.

    Deliberately not `--write --arm ms`: while the choice was open neither candidate could carry a
    release name, and a staged file must never be promoted with `mv` (the stem is written into the
    archive). Promotion was `--write --arm ms` once `ARMS` named the winner, which is what happened
    on 2026-09-10.

    Still here after the decision, because the comparison is the evidence for it: this writes both
    sides again, and neither file it writes is ever uploaded.
    """
    os.makedirs(out_dir, exist_ok=True)
    planned = [(c, t) for c in MS_CANDIDATES for t in tiers]
    existing = [(c, t) for c, t in planned
                if os.path.exists(os.path.join(out_dir, c.stem + _suffix(t)))]
    if existing and not force:
        print("refusing to overwrite:")
        for c, t in existing:
            print(f"  {os.path.join(out_dir, c.stem + _suffix(t))}")
        print("\n--force replaces them.")
        return 2

    print(f"staging both ms candidates into {out_dir}")
    print("neither carries a release name; the release files are written by --write.\n")
    for c in MS_CANDIDATES:
        n = os.path.getsize(c.src) if os.path.exists(c.src) else -1
        if n != SRC_BYTES:
            print(f"  MISSING or wrong size: {c.src} ({n:,} B)")
            return 2
    for c, t in planned:
        dst = os.path.join(out_dir, c.stem + _suffix(t))
        print(f"[{t:<8}] {c.stem}  val_loss {c.val_loss:.5f}  epoch {c.epoch}")
        print(f"  {c.note}")
        print(f"  src {c.src}")
        _emit(bda, c.src, dst, t)
        print(f"  sha256 {sha256(dst)}")
    return 0


def _suffix(tier):
    return ".pt" if tier == "research" else "_fp16.pt"


def do_manifest(bda, arms, tiers, out_dir):
    """The markdown table upload_plan.md quotes.

    Measured from disk. A row whose file is not there falls back to the recorded values and is
    marked, and it is left out of the total: a manifest that quietly totals files nobody has
    produced is exactly what an upload checklist must not do.
    """
    repos = {"research": "daylight-00/prepibind", "demo": "daylight-00/prepibind-demo"}
    for t in tiers:
        print(f"\n### {repos[t]} ({DTYPE[t]})\n")
        print("| file | bytes | MiB | md5 | sha256 | source | on disk |")
        print("|---|---:|---:|---|---|---|---|")
        total = missing = 0
        for a in arms:
            p = os.path.join(out_dir, a.filename(t))
            e = a.expect[t]
            here = os.path.exists(p)
            if here:
                n, m, s = os.path.getsize(p), bda.md5(p), sha256(p)
                total += n
            else:
                n, m, s = e.bytes, e.md5, e.sha256
                missing += 1
            mark = "yes" if here else ("**no - DISPUTED, not built**" if a.disputed
                                      else "**no - recorded values**")
            print(f"| `{a.filename(t)}` | {n:,} | {n / 1048576:.1f} | `{m}` | `{s}` | `{a.src}` "
                  f"| {mark} |")
        print(f"| **total on disk** | **{total:,}** | **{total / 1048576:.1f}** | | | | |")
        if missing:
            print(f"\n{missing} file(s) above are not on disk; their bytes/md5/sha256 are the "
                  f"recorded values and are excluded from the total.")
    return 0


def do_list(arms, tiers, out_dir):
    print(f"out-dir: {out_dir}\n")
    print(f"{'arm':<12} {'seed':<5} {'fold':<5} {'val_loss':>9} {'epoch':>6}  file")
    for a in arms:
        for t in tiers:
            p = os.path.join(out_dir, a.filename(t))
            mark = "present" if os.path.exists(p) else "absent "
            flag = "  <- DISPUTED, not written; see --stage-ms" if a.disputed else ""
            print(f"{a.name:<12} {a.seed:<5} {a.fold:<5} {a.val_loss:>9.5f} {a.epoch:>6}  "
                  f"[{mark}] {a.filename(t)}{flag}")
    print("\nsources:")
    return check_sources(arms) and 1 or 0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    m = ap.add_mutually_exclusive_group(required=True)
    m.add_argument("--write", action="store_true", help="produce the release files")
    m.add_argument("--check", action="store_true",
                   help="re-derive each into a scratch dir and byte-compare; writes nothing")
    m.add_argument("--manifest", action="store_true", help="the markdown table for upload_plan.md")
    m.add_argument("--verify", action="store_true",
                   help="load each release file into the model the predict config builds, "
                        "strict=True, and check the fp32/fp16 relation and the stored epoch")
    m.add_argument("--check-legacy", action="store_true",
                   help="re-derive each superseded models/prepi_esmc_small_*_fp16.pt under its own "
                        "old basename and byte-compare; this is the conversion-path proof")
    m.add_argument("--stage-ms", action="store_true",
                   help="write both ms candidates under staging names (the 2026-09-10 comparison)")
    m.add_argument("--list", action="store_true", help="what would be produced, and from what")
    ap.add_argument("--arm", action="append", choices=[a.name for a in ARMS],
                    help="restrict to one arm; repeatable (default: all four)")
    ap.add_argument("--tier", choices=("research", "demo", "both"), default="both")
    ap.add_argument("--out-dir", default=DEFAULT_OUT,
                    help=f"default {DEFAULT_OUT} (already git-ignored)")
    ap.add_argument("--tmp-dir", default=None,
                    help="where --check re-derives; needs 220 MB free (default: the system temp)")
    ap.add_argument("--force", action="store_true", help="with --write, replace existing files")
    ap.add_argument("--inspect-sources", action="store_true",
                    help="also load each source and print its structure (slow, reads 2.5 GB)")
    a = ap.parse_args()

    arms = [x for x in ARMS if not a.arm or x.name in a.arm]
    tiers = TIERS if a.tier == "both" else (a.tier,)
    out_dir = os.path.abspath(a.out_dir)

    #: A disputed arm never reaches --write or --check. Skipped silently would be worse than
    #: refused, so an explicit --arm on one is an error and a bare run says what it left out.
    disputed = [x for x in arms if x.disputed]
    if disputed and (a.write or a.check):
        if a.arm:
            print("refusing: the source checkpoint for these arms is not settled, so they have no "
                  "release name yet:")
            for x in disputed:
                print(f"  {x.name}")
            print("\n--stage-ms writes both ms candidates under staging names instead.")
            return 2
        arms = [x for x in arms if not x.disputed]
        print("skipping (source not settled, no release name): "
              + ", ".join(x.name for x in disputed))
        print("  --stage-ms writes both ms candidates under staging names.\n")

    if a.list:
        return do_list([x for x in ARMS if not a.arm or x.name in a.arm], tiers, out_dir)

    bda = load_builder()
    if not builder_supports_dtype(bda) and "research" in tiers and not a.list:
        print("note: demo/build_demo_assets.py has no --dtype switch, so the float32 tier uses the")
        print("      local fallback in _emit_float32. The demo tier still goes through "
              "cmd_checkpoint.\n")

    if a.stage_ms:
        return do_stage_ms(bda, tiers,
                           MS_STAGING if a.out_dir == DEFAULT_OUT
                           else os.path.join(out_dir, "ms_candidates"), a.force)

    if a.manifest:
        return do_manifest(bda, arms, tiers, out_dir)

    print("sources:")
    if check_sources(arms):
        print("\nsource checkpoints are wrong or unreadable; stopping.")
        return 2
    if a.inspect_sources:
        for x in arms:
            n, dt, numel, epoch = inspect_source(x.src)
            ok = "ok" if (n, dt, numel) == (64, ["torch.float32"], N_PARAMS) else "UNEXPECTED"
            print(f"  {x.name:<12} {n} tensors, {dt}, {numel:,} params, epoch {epoch}  [{ok}]")
    print()

    if a.verify:
        return do_verify(bda, arms, tiers, out_dir)
    if a.check_legacy:
        return do_check_legacy(bda, arms, out_dir, a.tmp_dir)
    if a.check:
        return do_check(bda, arms, tiers, out_dir, a.tmp_dir)
    return do_write(bda, arms, tiers, out_dir, a.force)


if __name__ == "__main__":
    sys.exit(main())
