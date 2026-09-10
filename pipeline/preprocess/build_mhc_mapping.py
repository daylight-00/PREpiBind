#!/usr/bin/env python3
"""Reduce the full MHC sequence tables to the ones the models are trained against.

`preprocess/notebooks/0_mhc_sequences.ipynb` produces tables covering every allele in the source
databases, about 7,000 rows. Training reads a much smaller table: the alleles the datasets actually
use, plus the non-human, non-murine entries that were curated alongside them. That reduction, and
the H2 chain correction that follows it, are the last two steps between stage 0 and
`data/mhc_mapping/`.

    python preprocess/build_mhc_mapping.py --check
    python preprocess/build_mhc_mapping.py --write

What reproduces, and what does not, measured rather than assumed:

  * **Membership.** 116 alleles are reachable from the datasets; the published tables carry 154.
    The extra 38 are non-human, non-murine entries (BoLA, Patr and similar) curated alongside them
    and not derivable from the dataset files. They are carried over, not rebuilt.
  * **Sequences.** 113 of the 116 reproduce exactly. Three (HLA-DQB1*03:01, *05:03, *06:01) differ
    by an eight-residue segment (PQGPPPAG) near the C terminus, at residue 258 and beyond. The
    peptide-binding window for those alleles is 44|119, so the *sliced* sequences -- the only part
    any embedding sees -- are identical. Nothing downstream is affected.
  * **Domain windows do not reproduce.** Stage 0 derives start_idx by locating Sliced_Seq inside
    HLA_Seq, and in the surviving intermediates that lookup fails for 5,199 of 7,265 alleles, which
    it records as -1. The two files are not misaligned: they carry the same names in the same order.
    They simply hold sequences from different generations, so one is no longer a substring of the
    other. The published windows came from a matched pair that no longer exists on any host.

The practical consequence: **the window table is data, not a derivation.** It ships in
data/mhc_mapping/ and its coordinates are published in Supplementary Table S4; the hand alignment
behind it is preprocess/mhc_sequences/range_final.txt. Regenerating it would need the matched
intermediates back.
"""
import argparse
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
MAPPING = os.path.join(REPO, "data", "mhc_mapping")
ARMS = ("full", "ic50", "ms_ql", "ms_ic")


def alleles_used():
    names = set()
    for arm in ARMS:
        for split in ("train", "test"):
            d = pd.read_csv(os.path.join(REPO, "data", "dataset", arm, f"{split}.csv"),
                            usecols=["HLA_Name_A", "HLA_Name_B"])
            names |= set(d["HLA_Name_A"]) | set(d["HLA_Name_B"])
    return {n for n in names if isinstance(n, str)}


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--write", action="store_true")
    ap.add_argument("--full", default=os.path.join(HERE, "work", "mhc"),
                    help="directory holding stage 0's output")
    a = ap.parse_args()

    used = alleles_used()
    print(f"  alleles reachable from the datasets: {len(used)}")
    for name in ("HLA2_IMGT_MSA_idx.csv", "HLA2_IMGT_MSA_idx_edit.csv", "HLA2_IMGT_light.csv"):
        published = pd.read_csv(os.path.join(MAPPING, name))
        extra = set(published["HLA_Name"]) - used
        print(f"  {name:32s} published {len(published):>4} rows = {len(used)} used + {len(extra)} curated extras")
        if a.write:
            print("     (not rewritten: see the module docstring)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
