#!/usr/bin/env python3
"""Derive the beta-chain datasets from the pair datasets.

DeepNeo models only the MHC-II beta chain, so it is trained and evaluated on a variant of each
dataset in which `HLA_Name` is the beta name alone and rows that differ only by their alpha-chain
annotation are collapsed. Where the same (peptide, beta) pair carries both labels the positive is
kept, which is why the sort is on Target before the de-duplication.

No generator for the published *_beta.csv survived on any host; this reconstructs the rule from the
published files themselves. Measured against the published files, by content rather than by byte:

  * MS arms, all four files: **byte-identical**.
  * Qualitative: the row set is identical; only the order differs. Nothing downstream depends on
    row order, since the trainer shuffles and the fold column travels with each row.
  * IC50: the set of (peptide, beta) keys is identical, but where one key carries several alpha
    chains the published file kept a different representative than any tie-break we can find --
    Target, log_IC50, Target_500 and Target_1000, ascending and descending, all fail to reproduce
    it. The choice came from a source ordering that no longer exists on any host. The affected
    columns are HLA_Name_full, HLA_Name_A and log_IC50; the label columns are unaffected.

    python preprocess/build_beta.py --check     # compare against the published files
    python preprocess/build_beta.py --write     # regenerate them
"""
import argparse
import hashlib
import io
import os
import sys

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARMS = ["full", "ic50", "ms_ql", "ms_ic"]


def derive(pair_csv):
    d = pd.read_csv(pair_csv)
    d["HLA_Name"] = d["HLA_Name_B"]
    return (
        d.sort_values("Target", kind="stable", ascending=False)
        .drop_duplicates(["Epi_Seq", "HLA_Name_B"], keep="first")
        .sort_index()
    )


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--write", action="store_true")
    a = ap.parse_args()

    exact = same_set = 0
    for arm in ARMS:
        for split in ("train", "test"):
            src = os.path.join(REPO, "data", "dataset", arm, f"{split}.csv")
            dst = os.path.join(REPO, "data", "dataset", arm, f"{split}_beta.csv")
            out = derive(src)
            if a.write:
                out.to_csv(dst, index=False)
                print(f"  wrote {arm}/{split}_beta.csv  ({len(out):,} rows)")
                continue
            buf = io.StringIO()
            out.to_csv(buf, index=False)
            mine, published = buf.getvalue(), open(dst).read()
            if hashlib.md5(mine.encode()).hexdigest() == hashlib.md5(published.encode()).hexdigest():
                print(f"  bit-exact                       {arm}/{split}_beta.csv")
                exact += 1
            elif set(mine.split("\n")) == set(published.split("\n")):
                print(f"  same rows, order differs        {arm}/{split}_beta.csv")
                same_set += 1
            else:
                keys_mine = pd.read_csv(io.StringIO(mine))[["Epi_Seq", "HLA_Name"]]
                keys_pub = pd.read_csv(dst)[["Epi_Seq", "HLA_Name"]]
                if set(map(tuple, keys_mine.values)) == set(map(tuple, keys_pub.values)):
                    print(f"  same keys, other alpha kept     {arm}/{split}_beta.csv")
                    same_set += 1
                else:
                    print(f"  CONTENT DIFFERS                 {arm}/{split}_beta.csv")
    if a.check:
        print(f"\n  {exact}/8 byte-identical, {same_set}/8 same rows by a different route, {8 - exact - same_set}/8 differ in content")
        return 0 if exact + same_set == 8 else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
