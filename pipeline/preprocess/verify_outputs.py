#!/usr/bin/env python3
"""Check the preprocessing output against the datasets published with the paper.

    python preprocess/verify_outputs.py

Exits non-zero on the first mismatch. Every stage here is CPU-deterministic, so the correct result
is byte equality, not a tolerance.
"""
import csv
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def main():
    rows = list(csv.DictReader(open(os.path.join(HERE, "expected_checksums.csv"))))
    bad = missing = 0
    for r in rows:
        p = os.path.join(REPO, r["path"])
        if not os.path.exists(p):
            print(f"  MISSING   {r['path']}")
            missing += 1
            continue
        got = md5(p)
        if got == r["md5"]:
            print(f"  ok        {r['path']}")
        else:
            print(f"  MISMATCH  {r['path']}\n              expected {r['md5']}\n              got      {got}")
            bad += 1
    print(f"\n{len(rows) - bad - missing}/{len(rows)} match, {bad} mismatched, {missing} missing")
    return 1 if (bad or missing) else 0


if __name__ == "__main__":
    sys.exit(main())
