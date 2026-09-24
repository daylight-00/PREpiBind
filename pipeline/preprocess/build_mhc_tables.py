#!/usr/bin/env python3
"""Rebuild the MHC sequence tables that this repository does not redistribute.

    python pipeline/preprocess/build_mhc_tables.py            # download and write
    python pipeline/preprocess/build_mhc_tables.py --check    # verify, write nothing

IPD-IMGT/HLA is CC BY-NoDerivs and asks that modified versions of its data not be redistributed
without permission. This repository therefore ships **no IPD-derived sequence**. What it ships is
the part that is ours: for every allele the models use, the four-field IPD-IMGT/HLA allele the
sequence was taken from, and the peptide-binding window coordinates. This script fetches release
3.59.0 from the database's own distribution, github.com/ANHIG/IMGTHLA, and rebuilds locally:

    data/mhc_mapping/HLA2_IMGT_light.csv          HLA_Name, HLA_Seq
    data/mhc_mapping/HLA2_IMGT_MSA_idx.csv        + Sliced_Seq, start_idx, end_idx
    data/mhc_mapping/HLA2_IMGT_MSA_idx_edit.csv   HLA_Seq with "|start|end" appended
    demo/data/mhc_mapping_demo.csv                the demo's window slices

All four are .gitignore'd. Run this once before the embedding pipeline, the demo or
analysis/figures/figS2_umap.py. Standard library only; it writes nothing outside
data/mhc_mapping/ and demo/data/.

## What ships, and what is rebuilt

data/mhc_mapping/mhc_sources.csv names the source of each of the 154 rows:

  * 116 `IPD-IMGT/HLA:3.59.0` -- rebuilt here. 114 name the exact four-field allele whose
    gap-stripped alignment row is the sequence the paper used, and reproduce byte for byte.
  * 18 `UniProt` -- murine H2. UniProt is CC BY 4.0, so these ship with their sequences in
    data/mhc_mapping/h2_uniprot.csv and are copied through unchanged.
  * 20 `IPD-MHC:not-redistributed` -- BoLA, Mamu and SLA. IPD-MHC carries the same NoDerivs terms
    and no code here produces them, so they are omitted. No dataset row references any of them,
    and the figure input that used to carry them already excluded them.

Two of the 116, HLA-DQB1*06:01 and HLA-DRB1*01:04, are marked `IPD-IMGT/HLA:revised-since`: their
sequences were revised in the database between the release the tables were cut from and 3.59.0, so
no allele at 3.59.0 reproduces them. Neither is referenced by any dataset row.

The rebuilt tables therefore carry 132 rows where the originals carried 154, and every allele
reachable from data/dataset/ -- 98 HLA and 18 H2 -- is present and identical.

## One column is recomputed rather than reproduced

Sliced_Seq is written as HLA_Seq[start_idx:end_idx]. Nothing in this repository reads that column:
analysis/figures/figS2_umap.py slices embeddings with start_idx/end_idx directly and the embedding
pipeline reads HLA_Seq. The original column was cut by hand against an earlier generation of the
sequences and does not follow that rule for every row.
"""
import argparse
import collections
import csv
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
MAPPING = os.path.join(REPO, "data", "mhc_mapping")
DEMO = os.path.join(REPO, "demo", "data")

sys.path.insert(0, HERE)
import fetch_mhc_alignment as fma  # noqa: E402  -- reuses its fetcher and alignment parser

RELEASE = fma.RELEASE


def load_csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path, header, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def sha256_rows(header, rows):
    h = hashlib.sha256()
    h.update((",".join(header) + "\n").encode())
    for r in rows:
        h.update((",".join(str(c) for c in r) + "\n").encode())
    return h.hexdigest()


def build(alignment):
    """mhc_sources.csv + h2_uniprot.csv + the alignment -> the four tables, as row lists."""
    sources = load_csv(os.path.join(MAPPING, "mhc_sources.csv"))
    h2 = {r["HLA_Name"]: r["HLA_Seq"] for r in load_csv(os.path.join(MAPPING, "h2_uniprot.csv"))}

    light, idx, edit, seqs = [], [], [], {}
    omitted = collections.defaultdict(list)
    for r in sources:
        name, src, allele = r["HLA_Name"], r["source"], r["source_allele"]
        if src.startswith("IPD-MHC"):
            omitted["IPD-MHC, not redistributed"].append(name)
            continue
        if src == "UniProt":
            seq = h2.get(name)
            if seq is None:
                omitted["missing from h2_uniprot.csv"].append(name)
                continue
        elif allele:
            if allele not in alignment:
                omitted[f"absent from release {RELEASE}"].append(name)
                continue
            seq = alignment[allele].replace("*", "")
        else:
            omitted["revised in IPD-IMGT/HLA since the tables were cut"].append(name)
            continue

        seqs[name] = seq
        start, end = r["start_idx"], r["end_idx"]
        windowed = start != "" and end != ""
        coords = r.get("edit_coords", "")
        light.append([name, seq])
        idx.append([name, seq, seq[int(start):int(end)] if windowed else "", start, end])
        edit.append([name, f"{seq}|{coords}" if coords else seq])

    demo = []
    for r in load_csv(os.path.join(DEMO, "demo_windows.csv")):
        seq = seqs.get(r["HLA_Name"])
        if seq is None:
            continue
        off, length = int(r["offset"]), int(r["length"])
        demo.append([r["HLA_Name"], seq[off:off + length]])
    return light, idx, edit, demo, omitted


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--release", default=RELEASE,
                    help=f"IPD-IMGT/HLA release branch in ANHIG/IMGTHLA (default {RELEASE} = 3.59.0)")
    ap.add_argument("--check", action="store_true", help="verify and write nothing")
    args = ap.parse_args()

    cache = os.path.join(HERE, "mhc_sequences", "_alignments")
    os.makedirs(cache, exist_ok=True)
    alignment = collections.OrderedDict()
    for loc in fma.LOCI:
        alignment.update(fma.parse_alignment(fma.fetch(loc, args.release, cache)))
    print(f"  {len(alignment)} four-field alleles at release {args.release}")

    light, idx, edit, demo, omitted = build(alignment)
    print(f"  rebuilt {len(light)} rows; demo mapping {len(demo)} rows")
    for why, names in sorted(omitted.items()):
        print(f"  omitted {len(names):>3}  {why}")
        if len(names) <= 4:
            print(f"           {', '.join(names)}")

    targets = [
        (os.path.join(MAPPING, "HLA2_IMGT_light.csv"), ["HLA_Name", "HLA_Seq"], light),
        (os.path.join(MAPPING, "HLA2_IMGT_MSA_idx.csv"),
         ["HLA_Name", "HLA_Seq", "Sliced_Seq", "start_idx", "end_idx"], idx),
        (os.path.join(MAPPING, "HLA2_IMGT_MSA_idx_edit.csv"), ["HLA_Name", "HLA_Seq"], edit),
        (os.path.join(DEMO, "mhc_mapping_demo.csv"), ["HLA_Name", "HLA_Seq"], demo),
    ]
    for path, header, rows in targets:
        print(f"  {os.path.relpath(path, REPO):44s} {len(rows):>4} rows  "
              f"sha256 {sha256_rows(header, rows)[:16]}")
        if not args.check:
            write_csv(path, header, rows)
    print("  --check: wrote nothing" if args.check else "  written")


if __name__ == "__main__":
    main()
