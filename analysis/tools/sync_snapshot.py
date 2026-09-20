#!/usr/bin/env python3
"""
Populate the 0_raw snapshot from the scratch experiment tree.

This is the step that used to be done by hand, which is how the snapshot ended up
both incomplete and lossy: several scratch directories share the same
`<lr>_<seed>` names, so flattening them into one level silently overwrote files.

The snapshot now mirrors the scratch path verbatim:

    0_raw/<scratch-relative-path>

Collisions are impossible by construction, and provenance is readable from the
path alone - `rp.source(rel)` is just `SCRATCH / rel`. `raw_manifest.csv` then only
has to record integrity (md5, size) and usage, not do any back-tracing.

    python tools/sync_snapshot.py --dry-run    # report what would change
    python tools/sync_snapshot.py              # copy into 0_raw/
    python tools/sync_snapshot.py --dest 0_raw.new
"""
from __future__ import annotations

import argparse
import fnmatch
import glob as globmod
import os
import shutil
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # plot_package/
SCRATCH = os.environ.get('IMG_SCRATCH', '')   # the scratch tree the snapshot mirrors

# Scratch experiment roots whose model outputs belong in the snapshot.
# Add a root here when a new experiment feeds a figure.
ROOTS = [
    '250513/1_bulk', '250513/2_lomo', '250513/4_lomo_chai', '250513/5_hum_ani',
    '250513/6_lomo_2', '250513/7_lomo_chai', '250513/8_lomo_etc', '250513/9_hum_ani_etc',
    '250516/7_ic50_etc', '250524/0_bulk', '250524/1_lomo', '250524/2_h2', '250524/4_ic50',
    '250527/1_ms_re', '250527/2_ms_esm3_re', '250529/1_lomo_beta', '250714/10_ms',
    # 260829 filled the runs that were never completed: MS at e5_s128 plus the one
    # esmc_small e5_s100 fold whose checkpoint existed, and LOMO deepneo e3_s128 for
    # the 30 molecules that seed's job never reached. Its drift_* directories are
    # measurement scratch, not results, and are deliberately not roots.
    '260829/ms', '260829/lomo',
    # 260830 filled tab:perf-summary's remaining blanks: af3, boltz, esmc_large,
    # esmc_medium and esm3_medium on MS (ms_ql) and IC50 (500/1000). Reached
    # through the 1_ms / 2_ic50 entries inside 260830/, which already held the
    # reference-tool rescoring. esm3_large sits alongside them and is filtered out
    # in the analysis configs, not here.
    '260830/1_ms', '260830/2_ic50',
    # 260905/9_blosum_rerun REPLACES the published BLOSUM62 cells. The published
    # baseline was fed full[2s:e] instead of full[s:e] - about half the groove
    # window - which contradicted Supplementary 94's shared preprocessing claim;
    # all 805 cells were recomputed with the fixed encoder. The old paths stay in
    # the snapshot and in this manifest so the published numbers remain
    # reproducible; the analysis configs are what switch over, by adding this root
    # and dropping blosum from the five published roots. Reached through the
    # 9_blosum_rerun entry inside 260905/, the same arrangement 260830 uses.
    '260905/9_blosum_rerun',
]

#: model-output files to snapshot from each root
OUTPUT_PATTERNS = ('pred-*', 'metrics-*')

#: per-fold ROC/PR diagnostic plots. Nothing in the analyses reads them and they can
#: be regenerated from pred-* plus the test sets, so they stay out of the snapshot.
EXCLUDE_PATTERNS = ('*_curve-*.png',)

#: analysis inputs, as scratch-relative globs. Without these the snapshot holds
#: predictions but no ground truth, so no metric can be computed off-cluster.
INPUT_GLOBS = [
    '250513/lomo/*/test.csv',
    '250529/lomo_beta/*/test.csv',
    '250513/5_hum_ani/ani_full.csv',
    '250519/template.csv',
    '250714/10_ms/template.csv',
    # The two 250519/2_anal_plot and 250714/10_ms metrics_results_max_.csv that
    # used to be listed here are gone. They were results precomputed under the old
    # best-seed-by-test-score rule, and figures read them directly instead of the
    # pipeline's own output - so two figures were drawn from numbers no config in
    # this repo could reproduce. The notebooks now read scoring/*/ like everything
    # else, and shipping the old files again would only invite that back.
    '250714/2_umap_test/HLA2_IMGT_MSA_idx.csv',
    # figures/4_upset.ipynb needs these three. They are not on cluster abc - the
    # dataset-organisation step ran on the alpha workstation. Copy them over and
    # re-run this script; until then the globs match nothing and say so.
    '250511/2_dataset/1_full_bal/hum_ani_full.csv',
    '250511/2_dataset/2_ic50/hum_ani_full.csv',
    '250520/1_dataset/3_ms_ql/hum_ani_full.csv',
]


def wanted() -> list[str]:
    """Scratch-relative paths that belong in the snapshot."""
    out: set[str] = set()

    for root in ROOTS:
        base = os.path.join(SCRATCH, root)
        if not os.path.isdir(base):
            print(f'  warning: scratch root missing: {root}', file=sys.stderr)
            continue
        for dirpath, _, files in os.walk(base):
            for f in files:
                if any(fnmatch.fnmatch(f, p) for p in EXCLUDE_PATTERNS):
                    continue
                if any(fnmatch.fnmatch(f, p) for p in OUTPUT_PATTERNS):
                    out.add(os.path.relpath(os.path.join(dirpath, f), SCRATCH))

    for g in INPUT_GLOBS:
        hits = globmod.glob(os.path.join(SCRATCH, g))
        if not hits:
            print(f'  warning: input glob matched nothing: {g}', file=sys.stderr)
        for p in hits:
            out.add(os.path.relpath(p, SCRATCH))

    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dest', default='0_raw', help='destination under plot_package/')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    dest = a.dest if os.path.isabs(a.dest) else os.path.join(HERE, a.dest)
    rels = wanted()

    copied = skipped = same = 0
    total_bytes = 0
    for rel in rels:
        src = os.path.join(SCRATCH, rel)
        dst = os.path.join(dest, rel)
        try:
            ssz = os.path.getsize(src)
        except OSError:
            skipped += 1
            continue
        total_bytes += ssz
        if os.path.exists(dst) and os.path.getsize(dst) == ssz:
            same += 1
            continue
        if not a.dry_run:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
        copied += 1

    print(f'\nsnapshot target: {dest}')
    print(f'  wanted        {len(rels):6d} files, {total_bytes / 1048576:.1f} MB')
    print(f'  already there {same:6d}')
    print(f'  {"would copy" if a.dry_run else "copied":13s} {copied:6d}')
    if skipped:
        print(f'  unreadable    {skipped:6d}')
    if not a.dry_run:
        print('\nnext: python tools/build_manifest.py')
    return 0


if __name__ == '__main__':
    sys.exit(main())
