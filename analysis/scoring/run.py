#!/usr/bin/env python3
"""Run one scoring analysis without its notebook.

    python run.py 8_strat

Analyses 1_whole .. 7_serotype_ms each have a notebook next to their config.py, and those
notebooks are how they were run. 8_strat has none, so this is the same call the notebooks make,
with nothing else around it. Outputs land in the analysis directory, which is what the pipeline
writes to by bare filename.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("analysis", help="directory under analysis/scoring, e.g. 8_strat")
    a = ap.parse_args()

    d = os.path.join(HERE, a.analysis)
    if not os.path.isfile(os.path.join(d, "config.py")):
        sys.exit(f"{a.analysis} has no config.py")

    os.chdir(d)
    sys.path.insert(0, HERE)
    sys.path.insert(0, d)
    import pipeline as pl
    from config import CONFIG

    _, _, rep_level = pl.run(CONFIG)
    print(rep_level.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
