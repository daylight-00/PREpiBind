#!/usr/bin/env python3
"""Bootstrap uncertainty on pooled Qualitative ROC-AUC -- tab:supp-bootstrap (S11).

Like 8_strat, this table had no code in the tree: it was computed ad hoc and could
not be re-derived when the BLOSUM62 baseline was re-run. It lives here now.

## Method, as the caption states it

1,000 resamples of the test rows. Within one resample every run of every
configuration is scored on the SAME resampled rows, then reduced exactly as
pipeline.aggregate() does -- 5 folds -> seed mean -> mean over 3 seeds -- so the
point estimate is the same number representation_level_results.csv reports.
Paired differences reuse the resample, which is what makes their CI narrower than
the difference of two marginal CIs.

DeepNeo is scored on test_beta.csv (47,666 rows) where the representations use
test.csv (48,352), so it cannot share resampled rows with them. It gets a marginal
CI and is absent from the paired block, exactly as the published table has it.

    python boot.py [--n 1000] [--seed 0]
"""
import argparse, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import pipeline as pl   # noqa: E402
import rawpath as rp    # noqa: E402

MODELS = ['blosum', 'deepneo', 'chai', 'esmc_small', 'esm3_small']
PAIRS = [('blosum', 'esm3_small'), ('blosum', 'esmc_small'), ('blosum', 'chai'),
         ('chai', 'esm3_small'), ('chai', 'esmc_small'), ('esm3_small', 'esmc_small')]


def fast_auc(y: np.ndarray, s: np.ndarray) -> float:
    """ROC-AUC by rank sum. Same value as sklearn; no per-call sort of the labels."""
    n1 = y.sum()
    n0 = len(y) - n1
    if n0 == 0 or n1 == 0:
        return np.nan
    r = pd.Series(s).rank().to_numpy()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    per = pd.read_csv(os.path.join(HERE, '..', '1_whole', 'per_run_results.csv'))
    per = per[per['model'].isin(MODELS)]
    # The written table carries relative paths so it does not depend on where the snapshot and
    # this checkout sit; resolve them the same way pipeline.py did when it produced the table.
    per['path'] = per['path'].map(rp.at)
    per['test_path'] = per['test_path'].map(rp.data)

    # predictions and labels, grouped by the test set they were scored against
    labels, preds = {}, {}
    for tp, d in per.groupby('test_path'):
        labels[tp] = pd.read_csv(tp)['Target'].to_numpy()
        for r in d.itertuples(index=False):
            preds.setdefault((r.model, tp), []).append(pl.read_pred(r.path, r.model))
    runs = {}     # model -> (test_path, list of (seed, pred))
    for tp, d in per.groupby('test_path'):
        for m, dm in d.groupby('model'):
            runs[m] = (tp, [(r.seed, pl.read_pred(r.path, r.model))
                            for r in dm.itertuples(index=False)])

    def reduce_runs(m, idx):
        """5 folds -> seed mean -> mean over seeds, on the resampled rows idx."""
        tp, rr = runs[m]
        y = labels[tp][idx]
        by_seed = {}
        for seed, p in rr:
            by_seed.setdefault(seed, []).append(fast_auc(y, p[idx]))
        return float(np.mean([np.mean(v) for v in by_seed.values()]))

    rng = np.random.default_rng(a.seed)
    sizes = {tp: len(v) for tp, v in labels.items()}
    draws = {m: [] for m in runs}
    for b in range(a.n):
        idx = {tp: rng.integers(0, n, n) for tp, n in sizes.items()}
        for m in runs:
            tp, _ = runs[m]
            draws[m].append(reduce_runs(m, idx[tp]))
        if (b + 1) % 100 == 0:
            print(f'  {b + 1}/{a.n}', flush=True)

    full = {m: reduce_runs(m, np.arange(sizes[runs[m][0]])) for m in runs}
    rows = []
    for m in MODELS:
        if m not in draws:
            continue
        d = np.array(draws[m])
        lo, hi = np.percentile(d, [2.5, 97.5])
        rows.append(dict(kind='marginal', a=m, b='', value=full[m], lo=lo, hi=hi))
    for x, y in PAIRS:
        if x not in draws or y not in draws:
            continue
        d = np.array(draws[x]) - np.array(draws[y])
        lo, hi = np.percentile(d, [2.5, 97.5])
        rows.append(dict(kind='paired', a=x, b=y, value=full[x] - full[y], lo=lo, hi=hi))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(HERE, 'bootstrap_results.csv'), index=False)
    pd.set_option('display.width', 160)
    print(out.to_string(index=False, float_format=lambda v: f'{v:+.4f}'))
    print(f'\nwrote bootstrap_results.csv  (n={a.n}, seed={a.seed})')


if __name__ == '__main__':
    main()
