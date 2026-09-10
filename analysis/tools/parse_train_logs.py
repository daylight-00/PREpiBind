#!/usr/bin/env python3
"""
Recover per-run validation metrics from the scratch train.log files.

Every training run logged, for each epoch, its validation loss and validation
ROC-AUC. code/train.py checkpoints at the lowest validation loss, so the metrics
at that epoch describe the model whose predictions are in the snapshot.

Those numbers matter because seed selection has to happen on something. Choosing
the seed by *test* ROC-AUC - what the notebooks did - is selection on the metric
being reported, which inflates it by an amount that varies per model. Choosing the
best of N training runs is entirely reasonable; it just has to be judged on the
validation split. This script recovers that judgement without retraining.

    python tools/parse_train_logs.py            # -> val_metrics.csv
    python tools/parse_train_logs.py --check    # coverage report only

Output columns:
    root mdir opt model fold   the training run
    val_loss val_auc          at the checkpointed (lowest val_loss) epoch
    epoch n_epochs            where that was, and how long the run went

Use val_loss, not val_auc. train.py defines its own roc_auc_score(outputs, labels)
and calls it as roc_auc_score(targets, outputs), so the logged Val ROC-AUC sorts by
the labels and treats the predictions as labels. The trapezoid is then taken over a
non-monotonic axis and the result is not an AUC at all: 21% of the values here fall
outside [0, 1], the largest being 27372. The function itself is correct when called
in its declared order - it matches sklearn exactly - so the fix is a one-line
argument swap, but every train.log written so far carries the broken value.

val_loss is unaffected, and is what train.py checkpoints on, so selecting a seed by
val_loss picks exactly the run the checkpoint logic already considered best.

Those out-of-range values are also why the epoch pattern has to accept a sign. It did
not: `Val ROC-AUC: (?P<va>[0-9.]+)` failed on every negative one and dropped the whole
line, so 47614 of 219161 epoch lines (21.7%, and 87% of one run's) never reached the
minimum, and the recorded val_loss was the minimum over whatever survived. Every value
below the checkpoint's own was invisible, so val_loss could only come out too high.
Fixed 2026-09-10; a line that still looks like an epoch line and does not parse is now
counted and reported rather than skipped.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRATCH = os.environ.get('IMG_SCRATCH', '')   # the scratch tree the snapshot mirrors
OUT = os.path.join(HERE, 'val_metrics.csv')

# Directories holding a train.log. These are finer-grained than sync_snapshot's
# ROOTS because 250513/1_bulk logs per numbered sub-run.
LOG_DIRS = [
    '250513/1_bulk/1', '250513/1_bulk/2', '250513/1_bulk/3', '250513/1_bulk/4',
    '250513/2_lomo', '250513/4_lomo_chai', '250513/5_hum_ani', '250513/6_lomo_2',
    '250513/7_lomo_chai', '250513/8_lomo_etc', '250513/9_hum_ani_etc',
    '250516/7_ic50_etc', '250524/0_bulk', '250524/1_lomo', '250524/2_h2',
    '250524/4_ic50', '250527/1_ms_re', '250527/2_ms_esm3_re', '250529/1_lomo_beta',
    # one log for both of 260829's roots, hence the shorter prefix match below
    '260829',
    # 260830 keeps a train.log per experiment dir, so both are listed
    '260830/1_ms', '260830/2_ic50',
]

#: Predictions produced by running inference against a checkpoint trained under a
#: different root. 260829 filled MS esmc_small e5_s100 fold4 from 250527's surviving
#: checkpoint, so its validation record lives in that experiment's log, not its own.
#: Consulted only after the prediction's own root has been tried and missed.
CROSS_ROOT = {'260829/ms': '250527/1_ms_re'}

#: Any number train.py can print. Its logging uses plain %.5f today, so the sign is
#: what actually matters here, but a metric is not owed a fixed format and an
#: exponent costs nothing to accept.
NUM = r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?'

LINE = re.compile(
    r'\[(?P<path>[^\]]+)\]-\[Epoch (?P<ep>\d+)/(?P<tot>\d+)\].*?'
    rf'Val Loss: (?P<vl>{NUM}), Val ROC-AUC: (?P<va>{NUM})'
)
#: What an epoch line looks like before any field is read, so a line LINE cannot
#: parse can be counted instead of silently dropped - the shape of the bug above.
EPOCH = re.compile(r'\]-\[Epoch \d+/\d+\]')
NAME = re.compile(r'(?P<model>.+)_fold(?P<fold>\d+)$')

FIELDS = ['root', 'mdir', 'opt', 'model', 'fold', 'val_loss', 'val_auc', 'epoch', 'n_epochs']


def parse() -> dict:
    """(root, models_dir, opt, model, fold) -> row at the lowest-val_loss epoch.

    The models_* directory has to stay in the key: under LOMO it names the held-out
    molecule, and dropping it would merge 47 separate runs into one.
    """
    best: dict[tuple, dict] = {}
    seen_lines = 0
    unparsed: dict[str, int] = {}
    for root in LOG_DIRS:
        path = os.path.join(SCRATCH, root, 'train.log')
        if not os.path.isfile(path):
            print(f'  warning: no train.log under {root}', file=sys.stderr)
            continue
        for line in open(path, errors='ignore'):
            m = LINE.search(line)
            if not m:
                if EPOCH.search(line):
                    unparsed[root] = unparsed.get(root, 0) + 1
                continue
            seen_lines += 1
            parts = m.group('path').split('/')
            if len(parts) < 3:
                continue
            nm = NAME.match(parts[-1])
            if not nm:
                continue
            key = (root, parts[-3], parts[-2], nm.group('model'), int(nm.group('fold')))
            vl = float(m.group('vl'))
            if key not in best or vl < best[key]['val_loss']:
                best[key] = {'root': root, 'mdir': parts[-3], 'opt': parts[-2],
                             'model': nm.group('model'),
                             'fold': int(nm.group('fold')), 'val_loss': vl,
                             'val_auc': float(m.group('va')),
                             'epoch': int(m.group('ep')), 'n_epochs': int(m.group('tot'))}
    print(f'  parsed {seen_lines} epoch lines -> {len(best)} runs', file=sys.stderr)
    for root, n in sorted(unparsed.items(), key=lambda x: -x[1]):
        print(f'  warning: {n} epoch lines under {root} did not parse', file=sys.stderr)
    return best


def match(rel: str, index: dict):
    """Find the validation record for one snapshot prediction path.

    A prediction lives at <root>/<pdir>/<opt>/pred-<model>_fold<N>.csv. Under LOMO
    the models directory mirrors it (plots_X <- models_X). Elsewhere a root has one
    or two models directories serving several prediction directories - the H2 runs
    write both plot/ and plot_ani/ from one model, and deepneo trains in a *beta*
    directory - so we fall back to the root's models dirs, preferring a beta one for
    deepneo.
    """
    parts = rel.split('/')
    nm = NAME.match(parts[-1][len('pred-'):-len('.csv')])
    if not nm:
        return None
    model, fold, opt = nm.group('model'), int(nm.group('fold')), parts[-2]
    pdir = parts[-3]
    roots = ['/'.join(parts[:cut]) for cut in (3, 2, 1)]
    roots += [CROSS_ROOT[r] for r in roots if r in CROSS_ROOT]
    for root in roots:
        cands = []
        if pdir.startswith('plots_'):
            cands.append('models' + pdir[len('plots'):])
        cands += sorted(index.get(root, ()), key=lambda d: ('beta' in d) != (model == 'deepneo'))
        for mdir in cands:
            hit = index.get(root, {}).get(mdir, {}).get((opt, model, fold))
            if hit:
                return hit
    return None


def coverage(best: dict) -> None:
    """How many snapshot predictions can be matched to a validation record."""
    raw = os.path.join(HERE, '0_raw')
    index: dict = {}
    for (r, md, o, m, f), row in best.items():
        index.setdefault(r, {}).setdefault(md, {})[(o, m, f)] = row
    hit = miss = 0
    missing_by_root: dict[str, int] = {}
    for dirpath, _, files in os.walk(raw):
        for f in files:
            if not f.startswith('pred-'):
                continue
            rel = os.path.relpath(os.path.join(dirpath, f), raw)
            if match(rel, index):
                hit += 1
            else:
                miss += 1
                k = '/'.join(rel.split('/')[:2])
                missing_by_root[k] = missing_by_root.get(k, 0) + 1
    print(f'  snapshot predictions with a validation record: {hit}, without: {miss}',
          file=sys.stderr)
    for k, v in sorted(missing_by_root.items(), key=lambda x: -x[1])[:6]:
        print(f'    {v:5d}  {k}', file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true', help='report coverage, write nothing')
    a = ap.parse_args()
    best = parse()
    coverage(best)
    if a.check:
        return 0
    with open(OUT, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(sorted(best.values(),
                            key=lambda r: (r['root'], r['mdir'], r['opt'], r['model'], r['fold'])))
    print(f'\n-> {OUT}  ({len(best)} rows)', file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
