#!/usr/bin/env python3
"""Run the whole preprocessing chain, IEDB export to the four dataset arms, in one command.

    python preprocess/run_all.py                 # everything
    python preprocess/run_all.py --from 2        # resume at stage 2
    python preprocess/run_all.py --list          # show the stages and stop

These are the notebooks the published datasets were built with. Nothing was rewritten; only the
hardcoded input paths were replaced with the names in paths.py, so the chain runs from a checkout.
Each stage executes in its own working directory because the notebooks write their tables by bare
filename, which is how they were run.

Stage order is a dependency order, not a preference: the two MS arms filter against the arm that
supplies their negatives, so 4 needs 2 and 5 needs 3.

Verify the result with verify_outputs.py; it compares md5 against expected_checksums.csv.
"""
import argparse
import os
import sys
import time

import paths as P

#          notebook                    workdir        reads the raw IEDB export?
#
# Only stage 1 does. The arms read draft.csv; their export reads are commented out in the
# notebooks, which is how they were run.
STAGES = [
    (0, "0_mhc_sequences.ipynb",  "mhc",         False, "IPD-IMGT/HLA and UniProt sequences, domain windows"),
    (1, "1_iedb_to_draft.ipynb",  "draft",       True,  "IEDB export -> draft.csv"),
    (2, "2_arm_qualitative.ipynb","full", False,  "Qualitative arm -> data/dataset/full"),
    (3, "3_arm_ic50.ipynb",       "ic50",        False, "IC50 arm -> data/dataset/ic50"),
    (4, "4_arm_ms_ql.ipynb",      "ms_ql",       False, "MS arm, Qualitative negatives (needs stage 2)"),
    (5, "5_arm_ms_ic.ipynb",      "ms_ic",       False, "MS arm, IC50 negatives (needs stage 3)"),
]
HERE = os.path.dirname(os.path.abspath(__file__))


def run(stage):
    n, nb, workdir, _needs_iedb, what = stage
    import nbformat
    from nbclient import NotebookClient

    src = os.path.join(HERE, "notebooks", nb)
    cwd = P.arm_dir(workdir)
    print(f"[{n}] {what}\n    {nb}  in  {cwd}", flush=True)
    t0 = time.time()
    doc = nbformat.read(src, as_version=4)
    # The notebooks import paths.py relative to their working directory, which only holds when the
    # scratch tree sits inside preprocess/. Put this directory on the kernel's path so PREPIBIND_WORK
    # can point anywhere.
    env = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = HERE + (os.pathsep + env if env else "")
    NotebookClient(doc, timeout=None, kernel_name="python3", resources={"metadata": {"path": cwd}}).execute()
    print(f"    done in {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="start", type=int, default=0, help="first stage to run")
    ap.add_argument("--only", type=int, help="run a single stage")
    ap.add_argument("--list", action="store_true", help="list the stages and exit")
    a = ap.parse_args()

    if a.list:
        for n, nb, _, _, what in STAGES:
            print(f"  {n}  {nb:26s} {what}")
        return 0

    todo = [s for s in STAGES if (s[0] == a.only if a.only is not None else s[0] >= a.start)]
    if any(s[3] for s in todo) and not os.path.exists(P.IEDB_EXPORT):
        print(f"IEDB export not found at {P.IEDB_EXPORT}\n"
              f"Set PREPIBIND_IEDB_EXPORT to the unzipped mhc_ligand_full.csv. See preprocess/README.md.",
              file=sys.stderr)
        return 2

    # Stage 0 reads an alignment this repository does not ship. Fail with the command, not a
    # FileNotFoundError from inside a notebook kernel.
    if any(s[0] == 0 for s in todo):
        alignment = os.path.join(P.MHC_SRC, "HLA2_IMGT.csv")
        if not os.path.exists(alignment):
            print(f"stage 0 needs {alignment}, which is fetched rather than shipped:\n"
                  f"    python pipeline/preprocess/fetch_mhc_alignment.py",
                  file=sys.stderr)
            return 2

    os.makedirs(P.WORK, exist_ok=True)
    for s in todo:
        run(s)
    print("\nAll stages finished. Now run:  python preprocess/verify_outputs.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
