#!/usr/bin/env python3
"""Correct two murine H2 chain assignments in the MHC sequence tables.

Two entries were wrong in the tables the models were trained against, and the correction is applied
here rather than upstream because it was found after the sequences were built:

  * H2-IAd had its alpha and beta chains swapped, and each was sliced with the other's domain
    window, so both were frame-shifted by about ten residues as well as being in the wrong slot.
  * H2-IAg7 held the I-A(b) alpha chain (P14434) where it should hold the I-A(d) alpha (P04228).
    NOD I-A(g7) is Aalpha(d) paired with Abeta(g7), so its alpha chain is the I-A(d) alpha.

Both were confirmed against UniProt and against PDB 6BLX, whose chain A is the A-D alpha. The
corrected sequence for both alpha keys is the same one, P04228, which the tables already carry under
H2-IAdB, so this is a reassignment rather than a new sequence. Domain windows are unchanged:
H2-IAdA and H2-IAg7A keep 29|110, H2-IAdB keeps 39|114.

    python preprocess/apply_h2_correction.py --check
    python preprocess/apply_h2_correction.py --write
"""
import argparse
import os
import sys

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAPPING = os.path.join(REPO, "data", "mhc_mapping")
TABLES = ["HLA2_IMGT_light.csv", "HLA2_IMGT_MSA_idx.csv", "HLA2_IMGT_MSA_idx_edit.csv"]

# target key -> key whose sequence it should hold
REASSIGN = {"H2-IAdA": "H2-IAdB", "H2-IAdB": "H2-IAdA", "H2-IAg7A": "H2-IAdB"}


def is_defective(seq):
    """The defect's signature: H2-IAg7A holds the I-A(b) alpha, so it equals H2-IAbA."""
    return seq["H2-IAg7A"] == seq["H2-IAbA"]


def corrected(df):
    seq = dict(zip(df["HLA_Name"], df["HLA_Seq"]))
    if not is_defective(seq):
        return df                      # already corrected; the reassignment is not idempotent
    out = df.copy()
    for target, source in REASSIGN.items():
        out.loc[out["HLA_Name"] == target, "HLA_Seq"] = seq[source]
    return out


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--write", action="store_true")
    a = ap.parse_args()

    bad = 0
    for name in TABLES:
        p = os.path.join(MAPPING, name)
        df = pd.read_csv(p)
        fixed = corrected(df)
        if a.write:
            fixed.to_csv(p, index=False)
            print(f"  wrote {name}")
            continue
        same = df["HLA_Seq"].equals(fixed["HLA_Seq"])
        print(f"  {'already corrected' if same else 'NEEDS CORRECTION '}  {name}")
        bad += not same
    if a.check:
        seq = dict(zip(*[pd.read_csv(os.path.join(MAPPING, TABLES[1]))[c] for c in ("HLA_Name", "HLA_Seq")]))
        print(f"\n  H2-IAg7A == H2-IAdA : {seq['H2-IAg7A'] == seq['H2-IAdA']}  (must be True)")
        print(f"  H2-IAg7A == H2-IAbA : {seq['H2-IAg7A'] == seq['H2-IAbA']}  (must be False)")
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
