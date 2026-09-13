#!/usr/bin/env python3
"""Fetch the IPD-IMGT/HLA class II protein alignment and rebuild the two tables stage 0 reads.

    python pipeline/preprocess/fetch_mhc_alignment.py            # download and write
    python pipeline/preprocess/fetch_mhc_alignment.py --check    # verify, write nothing

The alignment is not redistributed in this repository. IPD-IMGT/HLA is CC BY-NoDerivs and asks that
its data be linked to rather than mirrored, so this fetches it from the database's own distribution,
`github.com/ANHIG/IMGTHLA`, at the release the paper used, and rebuilds locally:

    mhc_sequences/MHC2MSA.csv    12,212 rows, four-field allele names, the alignment as published
    mhc_sequences/HLA2_IMGT.csv   7,267 rows, two-field names, what notebooks/0_mhc_sequences reads

Both are `.gitignore`d. Everything downstream of them -- `data/mhc_mapping/`, the embeddings, the
datasets -- is in the repository already, so this is only needed to re-derive stage 0.

Standard library only, and it writes nothing outside `mhc_sequences/`.

## How the rebuild works

IMGT ships one alignment per locus. In each file the first allele is the reference, `-` means
"same residue as the reference", `*` is an unsequenced position and `.` is an alignment gap.
Expanding the dashes and mapping `.` to `*` reproduces `MHC2MSA.csv` **byte for byte** (12,212/12,212
verified 2026-09-13 against release 3.59.0).

`HLA2_IMGT.csv` collapses those to two-field names: names with a `N` or `Q` suffix are dropped, a
group whose members all carry the same sequence keeps it, and the 296 groups that disagree take the
sequence recorded in `mhc_sequences/filtered_manual.json` -- the hand decision made when these
tables were first built. Everything else takes the most completely sequenced member (fewest `*`).

This one is not byte-identical to the copy it replaces, and both differences are accounted for.
All 7,267 rows of the retired file come back with the same residues; it writes 7,283, the extra 16
being two-field names the older generation did not have. And the retired copy writes `.` in 9,141
positions where 3.59.0 writes `*`, having been staged from an earlier release, with **zero**
differences in any actual residue -- this script normalises both gap characters to `*`, which is
what `MHC2MSA.csv` already did and what stage 0 strips anyway.

## What it does not reproduce

`data/mhc_mapping/` (154 rows, in the repository) was cut by hand before this chain existed, and 7
of its 116 HLA rows carry sequences that differ from release 3.59.0 in real residues -- a database
revision between generations, not an error here. `--check` names them. The other 109 reproduce
exactly. No published number depends on re-deriving them: the tables themselves are what was
trained on, and they ship.
"""
import argparse
import collections
import csv
import hashlib
import json
import os
import re
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
MHC_SRC = os.path.join(HERE, "mhc_sequences")
MAPPING = os.path.join(REPO, "data", "mhc_mapping", "HLA2_IMGT_MSA_idx.csv")

RELEASE = "3590"                      # IPD-IMGT/HLA 3.59.0, 2025-01-15: the release the paper used
BASE = "https://raw.githubusercontent.com/ANHIG/IMGTHLA/{rel}/alignments/{loc}_prot.txt"
LOCI = ["DPA1", "DPB1", "DQA1", "DQA2", "DQB1", "DQB2", "DRA", "DRB"]

# sha256 of the two rebuilt files at release 3590, recorded 2026-09-13 on abc.
EXPECTED = {
    "MHC2MSA.csv": "d492c29e5dec1df28bcbed2d8a8a9472d696182167398e42d466b8aa005f3927",
    "HLA2_IMGT.csv": "b99462ba0f7fbfc9f743a28e5e1d92f800da8523b957e9f55b9cf3a3badfcc04",
}

# The peptide-binding windows stage 0 slices, one per locus family. Kept here only so --check can
# re-cut them; the authoritative copy is data/mhc_mapping/.
WINDOWS = {"DPA": (35, 114), "DPB": (42, 121), "DQA": (29, 110),
           "DQB": (45, 119), "DRA": (30, 109), "DRB": (42, 121)}


def fetch(loc, release, cache_dir):
    """Download one locus alignment, or reuse an already-downloaded copy."""
    dest = os.path.join(cache_dir, f"{loc}_prot_{release}.txt")
    if os.path.exists(dest):
        return dest
    url = BASE.format(rel=release, loc=loc)
    with urllib.request.urlopen(url, timeout=120) as r:
        body = r.read()
    if body.startswith(b"404"):
        sys.exit(f"{url} returned 404 -- is {release} a released version?")
    with open(dest, "wb") as fh:
        fh.write(body)
    return dest


def parse_alignment(path):
    """IMGT alignment -> {allele: aligned sequence}, dashes expanded, '.' normalised to '*'."""
    blocks, order, ref = {}, [], None
    for line in open(path):
        if line.startswith("#") or not line.strip() or not line.startswith(" "):
            continue
        tokens = line.split()
        name = tokens[0]
        if not re.match(r"^[A-Z]+[A-Z0-9]*\*\d", name):      # ruler and "Prot" lines
            continue
        if name not in blocks:
            blocks[name], ref = "", ref or name
            order.append(name)
        blocks[name] += "".join(tokens[1:])
    reference = blocks[ref]
    out = collections.OrderedDict()
    for name in order:
        seq = blocks[name]
        expanded = "".join(reference[i] if c == "-" and i < len(reference) else c
                           for i, c in enumerate(seq))
        out["HLA-" + name] = expanded.replace(".", "*")
    return out


def two_field(name):
    locus, rest = name.split("*")
    fields = rest.split(":")
    return f"{locus}*{fields[0]}:{fields[1]}" if len(fields) >= 2 else name


def collapse(alignment, manual):
    """Four-field alignment -> one row per two-field name."""
    groups = collections.OrderedDict()
    for name in alignment:
        groups.setdefault(two_field(name), []).append(name)
    out = collections.OrderedDict()
    for short, members in groups.items():
        if re.search(r"[A-Z]$", short):        # N (null) and Q (questionable) expression
            continue
        if short in manual:
            out[short] = manual[short]
        else:
            out[short] = min((alignment[m] for m in members), key=lambda s: s.count("*"))
    return out


def write_csv(path, header, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")   # the retired tables are LF, not CRLF
        w.writerow(header)
        w.writerows(rows)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_mapping(collapsed):
    """Re-cut the published windows from the rebuild and report what still matches."""
    if not os.path.exists(MAPPING):
        return None
    exact, drift, skipped = 0, [], 0
    for row in csv.DictReader(open(MAPPING)):
        name = row["HLA_Name"]
        if name not in collapsed:                 # H2 (UniProt) and the IPD-MHC rows
            skipped += 1
            continue
        window = next((w for k, w in WINDOWS.items() if k in name), None)
        seq = collapsed[name]
        if (seq[window[0] - 1:window[1]].replace("*", "") == row["Sliced_Seq"]
                and seq.replace("*", "") == row["HLA_Seq"]):
            exact += 1
        else:
            drift.append(name)
    return exact, drift, skipped


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--release", default=RELEASE,
                    help=f"IPD-IMGT/HLA release branch in ANHIG/IMGTHLA (default {RELEASE} = 3.59.0)")
    ap.add_argument("--check", action="store_true", help="verify and write nothing")
    args = ap.parse_args()

    cache = os.path.join(MHC_SRC, "_alignments")
    os.makedirs(cache, exist_ok=True)
    alignment = collections.OrderedDict()
    for loc in LOCI:
        path = fetch(loc, args.release, cache)
        got = parse_alignment(path)
        print(f"  {loc:5s} {len(got):6d} alleles")
        alignment.update(got)
    print(f"  total {len(alignment)} four-field alleles at release {args.release}")

    manual_path = os.path.join(MHC_SRC, "filtered_manual.json")
    manual = {m["HLA_short"]: m["Sequence"][0].replace(".", "*")
              for m in json.load(open(manual_path))}
    collapsed = collapse(alignment, manual)
    print(f"  {len(collapsed)} two-field names after dropping N/Q and collapsing "
          f"({len(manual)} groups taken from filtered_manual.json)")

    out_dir = cache if args.check else MHC_SRC
    msa = os.path.join(out_dir, "MHC2MSA.csv")
    imgt = os.path.join(out_dir, "HLA2_IMGT.csv")
    write_csv(msa, ["HLA", "Sequence"], alignment.items())
    write_csv(imgt, ["HLA_Name", "HLA_Seq"], collapsed.items())

    ok = True
    for path in (msa, imgt):
        name = os.path.basename(path)
        digest = sha256(path)
        want = EXPECTED.get(name)
        if want and want.startswith("PLACEHOLDER"):
            print(f"  {name:16s} sha256 {digest}")
        elif digest == want:
            print(f"  {name:16s} sha256 matches")
        else:
            ok = False
            print(f"  {name:16s} sha256 {digest}\n  {'':16s} EXPECTED {want}")

    result = check_mapping(collapsed)
    if result:
        exact, drift, skipped = result
        print(f"\n  data/mhc_mapping/HLA2_IMGT_MSA_idx.csv: {exact} of {exact + len(drift)} HLA rows "
              f"re-cut exactly, {skipped} non-HLA rows skipped (H2 from UniProt, and IPD-MHC)")
        if drift:
            print("  sequence revised between the generation those rows were cut from and "
                  f"{args.release}:\n    " + ", ".join(drift))

    if args.check:
        print(f"\n  --check: wrote nothing outside {os.path.relpath(cache, REPO)}/")
    else:
        print(f"\n  wrote {os.path.relpath(msa, REPO)} and {os.path.relpath(imgt, REPO)}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
