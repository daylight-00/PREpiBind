#!/usr/bin/env python3
"""Build the demo's two artifacts from the ones the paper used.

    python demo/build_demo_assets.py hla-store  --check | --write
    python demo/build_demo_assets.py checkpoint <run>/models/e5_s100/esmc_small_fold0-best.pt <out>
                                     [--dtype float16|float32]

The demo is not the paper. It encodes epitopes with ESMC at run time instead of reading a
precomputed store, so it cannot match the published numbers bit for bit however carefully it is
built. Given that, it takes the memory and the download time instead: half precision throughout,
the peptide-binding window cut out ahead of time, and only the alleles the datasets use.

Nothing here is a separate measurement. Both outputs are pure functions of what
`pipeline/embeddings/` and training already produced, so `--check` can prove it.

  hla-store   full-length float32 store  -> window sliced out, cast to float16, 116 alleles
  checkpoint  training checkpoint        -> model weights only, cast to `--dtype`

`hla-store` reads the research store; PREPIBIND_EMB_ROOT names the directory holding it. There is
no default, because a fresh clone has the store nowhere. `checkpoint` needs no store.

The 116 alleles are the ones reachable from `data/dataset/`; the demo's own dataset uses 110 of
them. The research store carries 154, the extra 38 being the non-human entries curated alongside
them, which no released model was trained to score.
"""
import argparse
import glob
import hashlib
import os
import sys

import h5py
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

RESEARCH_STORE_NAME = "emb_hla_esmc_small_0430.h5"
WINDOWS = os.path.join(REPO, "data", "mhc_mapping", "HLA2_IMGT_MSA_idx_edit.csv")
OUT_STORE = os.path.join(HERE, "data", "emb_hla_esmc_small_demo_fp16.h5")
OUT_MAP = os.path.join(HERE, "data", "mhc_mapping_demo.csv")


def research_store():
    """The full-length float32 HLA store this script slices.

    It is not distributed with the repository, so no path in a fresh clone holds it and there is no
    default: PREPIBIND_EMB_ROOT must name the directory it is in. Resolved on use, not at import,
    so that importing this module for its checkpoint helpers does not require a store.
    """
    root = os.environ.get("PREPIBIND_EMB_ROOT")
    if not root:
        sys.exit("PREPIBIND_EMB_ROOT is not set. Point it at the directory holding "
                 f"{RESEARCH_STORE_NAME}; it is not distributed with this repository. It is the "
                 "HuggingFace dataset daylight-00/prepibind-embeddings, and "
                 "pipeline/embeddings/ is the code that produced it.")
    return os.path.join(root, RESEARCH_STORE_NAME)


def alleles_used():
    """Every allele name the four dataset arms refer to, alpha and beta."""
    names = set()
    for f in sorted(glob.glob(os.path.join(REPO, "data", "dataset", "*", "*.csv"))):
        d = pd.read_csv(f)
        for col in ("HLA_Name_A", "HLA_Name_B"):
            if col in d.columns:
                names |= {str(x) for x in d[col].dropna()}
    return names


def windows():
    """allele -> (start, end). The table stores it as `sequence|start|end`."""
    from prepibind.encoder import split_hla
    out = {}
    for _, r in pd.read_csv(WINDOWS).iterrows():
        seq, start, end = split_hla(r["HLA_Seq"])
        out[r["HLA_Name"]] = (seq, start, end)
    return out


def build_store():
    """(allele -> float16 window) and the mapping table that goes with it."""
    keep = alleles_used()
    win = windows()
    missing = sorted(k for k in keep if k not in win)
    if missing:
        sys.exit(f"no window for {len(missing)} alleles: {missing[:5]}")

    arrays, rows = {}, []
    store = research_store()
    with h5py.File(store, "r") as f:
        absent = sorted(k for k in keep if k not in f)
        if absent:
            sys.exit(f"{len(absent)} alleles are not in {store}: {absent[:5]}")
        for name in sorted(keep):
            seq, start, end = win[name]
            emb = np.squeeze(f[name][()])
            if end > len(emb):
                sys.exit(f"{name}: window {start}:{end} does not fit a {len(emb)}-residue embedding")
            arrays[name] = emb[start:end].astype(np.float16)
            rows.append({"HLA_Name": name, "HLA_Seq": seq[start:end]})
    # No window columns: the store is already cut to it, and encoder.split_hla slices only when
    # the table asks it to. Adding them back would make the demo slice twice.
    return arrays, pd.DataFrame(rows)


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_hla_store(a):
    arrays, table = build_store()
    total = sum(v.nbytes for v in arrays.values())
    print(f"  {len(arrays)} alleles, {total / 1048576:.1f} MB of float16 windows")

    if a.check:
        bad = 0
        if not os.path.exists(OUT_STORE):
            print(f"  missing: {OUT_STORE}")
            return 1
        with h5py.File(OUT_STORE, "r") as f:
            have = set(f.keys())
            if have != set(arrays):
                print(f"  key sets differ: {len(have - set(arrays))} extra, {len(set(arrays) - have)} missing")
                bad += 1
            for k in sorted(have & set(arrays)):
                if not np.array_equal(np.squeeze(f[k][()]), arrays[k]):
                    bad += 1
        shipped = pd.read_csv(OUT_MAP)
        if not shipped.equals(table):
            print("  mapping table differs")
            bad += 1
        print(f"  {'ok' if not bad else str(bad) + ' differences'}")
        return 1 if bad else 0

    with h5py.File(OUT_STORE, "w") as f:
        for k, v in arrays.items():
            f.create_dataset(k, data=v)
    table.to_csv(OUT_MAP, index=False)
    print(f"  -> {OUT_STORE}  ({os.path.getsize(OUT_STORE) / 1048576:.1f} MB, md5 {md5(OUT_STORE)})")
    print(f"  -> {OUT_MAP}  ({len(table)} rows)")
    return 0


#: `--dtype` -> the cast applied to every floating-point tensor. float16 is the default because
#: it is what every fp16 checkpoint shipped so far came out of; sources are float32 already, so
#: float32 is a pure strip (optimizer state and epoch dropped, values untouched).
CASTS = {"float16": lambda t: t.half(), "float32": lambda t: t.float()}


def cmd_checkpoint(a):
    import torch
    dtype = getattr(a, "dtype", None) or "float16"
    if dtype not in CASTS:
        sys.exit(f"unknown --dtype {dtype!r}; choose one of {', '.join(sorted(CASTS))}")
    cast = CASTS[dtype]
    ck = torch.load(a.src, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]
    out = {k: cast(v) if v.is_floating_point() else v for k, v in sd.items()}
    n = sum(v.numel() for v in out.values())
    torch.save({"model_state_dict": out}, a.dst)
    print(f"  {n / 1e6:.2f} M parameters, epoch {ck.get('epoch')}")
    print(f"  {os.path.getsize(a.src) / 1048576:.1f} MB -> {os.path.getsize(a.dst) / 1048576:.1f} MB"
          f"  (optimizer state dropped, weights cast to {dtype})")
    print(f"  -> {a.dst}  md5 {md5(a.dst)}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("hla-store", help="the sliced float16 HLA store and its mapping table")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--write", action="store_true")
    s.set_defaults(fn=cmd_hla_store)

    c = sub.add_parser("checkpoint", help="a training checkpoint, stripped and cast")
    c.add_argument("src")
    c.add_argument("dst")
    c.add_argument("--dtype", choices=sorted(CASTS), default="float16",
                   help="precision of the emitted weights (default: float16, the demo tier)")
    c.set_defaults(fn=cmd_checkpoint)

    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
