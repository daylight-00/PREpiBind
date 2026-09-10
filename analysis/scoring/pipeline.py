"""
Shared pipeline for the per-analysis prediction reviews under scoring/.

Every analysis (1_whole, 2_ms, 3_ic, 4_lomo, 5_h2, 6_serotype, 7_serotype_ms)
used to carry its own copy of this logic, which is how the same bug ended up fixed
in some copies and not others. The logic now lives here once; each analysis supplies
only its own configuration.

Reads exclusively through `rawpath`, so it runs on clusters that have no scratch
data. Provenance stays available via rp.source() / rp.find() and the src_root /
src_path columns that collect() carries through.

Aggregation
-----------
Nothing selects a seed. Every (seed, fold) run a representation has is kept, and
the reduction is strictly hierarchical:

    5 folds  --mean-->  1 value per seed  --mean +- SD-->  representation

The SD is therefore the spread across three independent training runs, computed
from three numbers with ddof=1. It is *not* the SD of all 15 fold/seed values:
folds share a test set and are not independent replicates, so pooling them would
understate the variance that matters and inflate any downstream test's df.

Allele-wise and LOMO results reduce the same way, ending at one value per allele
or per withheld molecule. That value - not the individual run - is the unit for
plots and for statistics, which is what keeps Wilcoxon from pseudo-replicating.

Checkpoint selection within a run is whatever train.py already did: the epoch with
the lowest validation loss. The Val ROC-AUC in train.log cannot be used for this
(train.py calls its own roc_auc_score with the arguments swapped, so 21% of the
logged values fall outside [0,1]); val_loss is unaffected and is the criterion the
saved checkpoints were chosen by. See tools/parse_train_logs.py.

Typical use
-----------
    import sys; sys.path.insert(0, '..')
    import pipeline as pl
    from config import CONFIG

    per_run, seed_level, rep_level = pl.run(CONFIG)
"""
from __future__ import annotations

import fnmatch
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, auc, f1_score, matthews_corrcoef,
                            precision_recall_curve, roc_auc_score)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rawpath as rp  # noqa: E402

PRED_RE = re.compile(r'pred-(?P<model>.+)_fold(?P<fold>\d+)\.csv$')
VAL_METRICS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'val_metrics.csv')
#: models whose raw output is already a probability, so no sigmoid is applied
NO_SIGMOID = ('deepneo',)

#: the two published tools, as (template model id) -> (prediction file, %Rank column).
#: Produced by scoring/0_ref/ (prep -> score -> collect); see
#: docs/notes/2026-08-30-ref-rescore.md. Both columns are a %Rank against a fixed
#: random-peptide background, lower meaning better binder, so the score fed to a
#: metric is -%Rank. It is the only *comparable* quantity the two tools share.
#: MixMHC2pred-2.0 does emit raw scores under its -e flag (Score_, ScorePWM_),
#: which 0_ref/score.py does not pass: none of them is an affinity, its README
#: warns Score_ cannot be transformed to the %Rank, and ScorePWM_ is unbounded
#: and PWM-only, so none is on a shared scale with NetMHCIIpan's Score_EL.
#: See docs/notes/2026-09-01-mixmhc2pred-no-affinity-head.md.
#:
#: The column here is a *default*. An analysis overrides it with a `ref_col` dict
#: in its config.py, because NetMHCIIpan-4.3 has two heads with different training
#: targets and the right one depends on what the test set measures:
#:
#:     EL (el_rank, el_score)      eluted ligand / immunopeptidomics -> ms_ql
#:     BA (ba_rank, ba_score, ba)  IEDB measured IC50                -> ic50
#:
#: 3_ic overrides to ba_rank for exactly that reason; scoring binarised measured
#: IC50 with the eluted-ligand head cost NetMHCIIpan 0.18-0.20 ROC-AUC, in the
#: direction that flatters the representations. See
#: docs/notes/2026-09-01-netmhciipan-el-vs-ba.md. Every override must stay a
#: %Rank so the -%Rank convention and the within-allele calibration caveat hold
#: unchanged.
REF_PRED = {
    'netmhcpan':  ('260830/ref/pred_netmhcpan.csv',  'el_rank'),
    'mixmhcpred': ('260830/ref/pred_mixmhcpred.csv', 'rank'),
}
#: ROC-AUC and PR-AUC only. Both are rank statistics, so they mean the same thing
#: on a %Rank as on a probability. F1 / accuracy / MCC do not: the
#: representations are thresholded at 0.5 on a sigmoid, and 0.5 on a percentile
#: rank is a very strong binder rather than a coin flip, so applying the same
#: number to both sides would score the tools at a threshold nobody proposes for
#: them. The hand-entered rows this replaces also carried only these two.
REF_METRICS = ['roc_auc', 'pr_auc']
METRICS = ['roc_auc', 'pr_auc', 'f1', 'accuracy', 'mcc']
#: the seeds every representation is supposed to have
SEEDS = ('s42', 's100', 's128')

#: 260905/9_blosum_rerun holds the BLOSUM62 baseline recomputed with the fixed
#: encoder. The published cells were fed full[2s:e] instead of full[s:e] -- about
#: half the groove window -- which contradicted Supplementary 94's shared
#: preprocessing claim, so all 805 were re-run. The published prediction files stay
#: in the snapshot and in raw_manifest.csv, because the old numbers must remain
#: reproducible; the switch therefore happens HERE, not by overwriting them.
#: Measured effect: allele-wise 0.780 -> 0.781, paired Wilcoxon p = 0.99. See
#: docs/notes/2026-09-06-blosum-rerun-result-is-null.md.
BLOSUM_RERUN_ROOT = '260905/9_blosum_rerun'


def use_rerun_blosum(df: pd.DataFrame) -> pd.DataFrame:
    """Keep blosum rows only from the re-run root; every other model is untouched.

    Every analysis that has a blosum row calls this, so no analysis can silently
    end up mixing the two encoders -- which would put 34-residue and 75-residue
    cells in one column with nothing on the page saying so.
    """
    if 'blosum' not in set(df['model']):
        return df
    keep = (df['model'] != 'blosum') | df['root'].str.startswith(BLOSUM_RERUN_ROOT)
    out = df[keep]
    if not (out['model'] == 'blosum').any():
        raise ValueError(
            f'use_rerun_blosum() dropped every blosum row: no root starting with '
            f'{BLOSUM_RERUN_ROOT!r} was collected. Add it to this config\'s roots.')
    return out.reset_index(drop=True)


# ------------------------------------------------------------------ collect
def collect(roots, dir_filter='plots*', test_for=None) -> pd.DataFrame:
    """Enumerate prediction files in the snapshot under one or more scratch roots.

    Since the snapshot mirrors the scratch tree, the metadata comes straight from
    the path:

        <root>/<dir>/<lr>_<seed>/pred-<model>_fold<N>.csv

    Columns:

        model lr seed fold   parsed from the path
        dir                  the directory under the root: plots, plots_500,
                             plots_ql, plots_<held-out molecule>, models_<...>
        kind                 'plots' or 'models', the prefix of `dir`
        group                <dir> minus that prefix, i.e. the held-out molecule
                             for LOMO-style layouts, else ''
        root                 scratch root this file came from
        rel path src_path    snapshot rel, openable path, original scratch path
        test_path            filled in by the test_for callable, if given

    `dir_filter` is a glob, or a sequence of globs, on the `dir` component. It
    defaults to 'plots*'.

    Passing 'models_*' as well is not a mistake: a handful of runs wrote their
    predictions next to their checkpoints instead of into plots_. 4_lomo does this
    to reach the s42 runs of chai and esmc_small, which exist nowhere else. Any
    (model, lr, seed, fold, group) present under both prefixes is deduplicated in
    favour of plots_, so widening the filter can only add cells, never change one.
    """
    if isinstance(roots, str):
        roots = [roots]
    if isinstance(dir_filter, str) or dir_filter is None:
        dir_filter = [dir_filter] if dir_filter else None

    frames = []
    for root in roots:
        t = rp.table(root=root)
        t = t[t['rel'].str.contains('/pred-')]
        if t.empty:
            raise ValueError(
                f'no prediction files under 0_raw/{root}/ - is the root spelled '
                f'correctly, and is it listed in tools/sync_snapshot.py?'
            )
        t = t.copy()
        t['root'] = root
        frames.append(t)
    t = pd.concat(frames, ignore_index=True)

    rows = []
    for r in t.itertuples(index=False):
        parts = r.rel.split('/')
        m = PRED_RE.search(parts[-1])
        if not m or len(parts) < 3:
            continue
        d = parts[-3]
        if dir_filter and not any(fnmatch.fnmatch(d, g) for g in dir_filter):
            continue
        kind = 'models' if d.startswith('models') else 'plots'
        lr, _, seed = parts[-2].partition('_')
        rows.append({
            'model': m.group('model'), 'lr': lr, 'seed': seed,
            'fold': int(m.group('fold')),
            'dir': d, 'kind': kind,
            'group': d[len(kind) + 1:] if d.startswith(kind + '_') else '',
            'root': r.root, 'rel': r.rel, 'path': r.path, 'src_path': r.src_path,
            # kept so the plotting cells that still say row.file_path keep working
            'file_path': r.path,
        })
    if not rows:
        raise ValueError(f'dir_filter={dir_filter!r} matched nothing under {roots}')
    df = pd.DataFrame(rows)
    df = _drop_shadowed_models_rows(df)
    df = attach_val(df)
    if test_for is not None:
        df['test_path'] = [test_for(r) for r in df.itertuples(index=False)]
    return df.sort_values(['model', 'lr', 'seed', 'dir', 'fold']).reset_index(drop=True)


def _drop_shadowed_models_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep a models_* row only when no plots_* row covers the same run."""
    if not (df['kind'] == 'models').any():
        return df
    key = ['model', 'lr', 'seed', 'fold', 'group']
    seen = set(map(tuple, df.loc[df['kind'] == 'plots', key].values))
    drop = df['kind'].eq('models') & df[key].apply(lambda r: tuple(r) in seen, axis=1)
    return df[~drop].reset_index(drop=True)


def attach_val(df: pd.DataFrame) -> pd.DataFrame:
    """Add val_auc / val_loss from val_metrics.csv, recovered from the training logs.

    Nothing selects on these any more - they are carried through so a run's
    validation behaviour stays inspectable next to its test score. Missing silently
    leaves the columns empty.
    """
    if not os.path.exists(VAL_METRICS):
        df['val_auc'] = np.nan
        df['val_loss'] = np.nan
        return df
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        '_ptl', os.path.join(os.path.dirname(VAL_METRICS), 'tools', 'parse_train_logs.py'))
    ptl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ptl)

    val = pd.read_csv(VAL_METRICS)
    index: dict = {}
    for r in val.itertuples(index=False):
        index.setdefault(r.root, {}).setdefault(r.mdir, {})[(r.opt, r.model, r.fold)] = r
    hits = [ptl.match(rel, index) for rel in df['rel']]
    df['val_auc'] = [h.val_auc if h is not None else np.nan for h in hits]
    df['val_loss'] = [h.val_loss if h is not None else np.nan for h in hits]
    return df


# ------------------------------------------------------------------ metrics
def read_pred(path: str, model: str) -> np.ndarray:
    """Read one prediction file.

    The files have no header, so header=None is mandatory; without it pandas eats
    the first prediction as a column name and shifts every row by one.
    """
    v = pd.read_csv(path, header=None, names=['Predicted'])['Predicted'].values
    if model not in NO_SIGMOID:
        v = 1.0 / (1.0 + np.exp(-v))
    return v


def calculate_metrics(y_true, y_score) -> dict:
    y_label = (np.asarray(y_score) >= 0.5).astype(int)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return {
        'roc_auc': roc_auc_score(y_true, y_score),
        'pr_auc': auc(recall, precision),
        'f1': f1_score(y_true, y_label),
        'accuracy': accuracy_score(y_true, y_label),
        'mcc': matthews_corrcoef(y_true, y_label),
    }


def add_metrics(df: pd.DataFrame, target_for=None, subsets=None,
                out: str | None = None, strict: bool = True) -> pd.DataFrame:
    """Score every row of collect()'s output. Test sets are cached per path.

    `target_for(row)` returns the label column to use, defaulting to 'Target'
    (3_ic switches between Target_500 and Target_1000 by directory).
    `subsets` is an optional list; each entry selects a slice of the test set that
    is scored in addition to the whole, emitting one output row per subset with a
    'sero' column naming it. An entry is either

        'DR'                     a substring of HLA_Name        (serotype analyses)
        ('No 9-mer overlap', fn) a name and fn(test_frame) -> boolean mask
                                 (the 9-mer-overlap strata, 8_strat)

    The callable form exists because the strata are a property of the epitope and
    its relation to the training set, not a substring of the allele name.

    A prediction file whose length does not match its test set is an error, not
    something to quietly truncate: the old code merged on index, which is an inner
    join, so a mismatched pair silently scored on whatever rows happened to line up.
    """
    cache: dict[str, pd.DataFrame] = {}
    rows, failed = [], []
    for r in df.itertuples(index=False):
        try:
            if r.test_path not in cache:
                cache[r.test_path] = pd.read_csv(r.test_path)
            test = cache[r.test_path]
            pred = read_pred(r.path, r.model)
            if len(pred) != len(test):
                raise ValueError(
                    f'{len(pred)} predictions vs {len(test)} test rows - '
                    f'this prediction was not made against {os.path.basename(r.test_path)}'
                )
            tcol = target_for(r) if target_for else 'Target'
            frame = test.assign(_pred=pred)
            for sero in (subsets or [None]):
                if sero is None:
                    sub, name = frame, None
                elif isinstance(sero, str):
                    sub, name = frame[frame['HLA_Name'].str.contains(sero)], sero
                else:
                    name, mask_fn = sero
                    sub = frame[mask_fn(frame)]
                rec = {**r._asdict()}
                if sero is not None:
                    rec['sero'] = name
                rec.update(calculate_metrics(sub[tcol].values, sub['_pred'].values))
                rows.append(rec)
        except Exception as e:                                   # noqa: BLE001
            failed.append((r.rel, str(e)))
    if failed:
        msg = f'{len(failed)} rows failed, e.g. ' + '; '.join(
            f'{a}: {b}' for a, b in failed[:3])
        if strict:
            raise RuntimeError(msg)
        print('warning:', msg)
    res = pd.DataFrame(rows)
    if out:
        res.to_csv(out, index=False)
    return res


# ------------------------------------------------------------------ aggregation
def pin_one_lr(met: pd.DataFrame, lr_filter=None) -> pd.DataFrame:
    """Reduce to a single learning rate per model, and prove that it worked.

    The learning rate has to be fixed *before* aggregating, and it has to be fixed
    on something other than the test metric. The old code took the best lr by test
    roc_auc after averaging, which is the same selection-on-the-reported-metric
    problem that seed selection had, just one level up.
    """
    out = lr_filter(met) if lr_filter is not None else met
    extra = (out.groupby('model')['lr'].nunique() > 1)
    if extra.any():
        bad = {m: sorted(out.loc[out['model'] == m, 'lr'].unique())
               for m in extra[extra].index}
        raise ValueError(
            f'more than one learning rate survives for {bad}. Pin it in the '
            f'analysis config (lr_filter) - picking one by test score here would '
            f'reintroduce selection on the reported metric.')
    return out.reset_index(drop=True)


def aggregate(per_run: pd.DataFrame, group_keys=(), metrics=METRICS):
    """folds -> seed -> representation, per the two-step rule.

    Returns (seed_level, rep_level).

    seed_level  one row per (model, lr, seed, *group_keys): the mean over that
                seed's folds, plus n_folds.
    rep_level   one row per (model, lr, *group_keys): mean and SD over the
                seed-level values, plus n_seeds and the seeds actually present.

    SD is the sample SD (ddof=1) of the per-seed means. With the usual three seeds
    that is three numbers; a group with one seed gets NaN, which is correct and
    visible rather than a silent zero.
    """
    gk = list(group_keys)
    seed_keys = ['model', 'lr', 'seed'] + gk
    rep_keys = ['model', 'lr'] + gk

    seed_level = per_run.groupby(seed_keys, as_index=False)[metrics].mean()
    seed_level['n_folds'] = (per_run.groupby(seed_keys)['fold']
                             .nunique().reset_index(drop=True).values)

    g = seed_level.groupby(rep_keys, as_index=False)
    rep_level = g[metrics].mean()
    sd = g[metrics].std(ddof=1)
    for m in metrics:
        rep_level[f'{m}_sd'] = sd[m].values
    # merged rather than assigned positionally: a SeriesGroupBy aggregation does not
    # always come back shaped like the frame above (it depends on the key set), and
    # assigning .values would then silently misalign the seed counts
    seeds_info = (seed_level.groupby(rep_keys)['seed']
                  .agg(n_seeds='nunique', seeds=lambda s: '|'.join(sorted(set(s))))
                  .reset_index())
    rep_level = rep_level.merge(seeds_info, on=rep_keys, how='left')

    order = rep_keys + [c for m in metrics for c in (m, f'{m}_sd')] + ['n_seeds', 'seeds']
    return seed_level, rep_level[order]


def coverage(per_run: pd.DataFrame, group_keys=(), seeds=SEEDS, n_folds=5) -> pd.DataFrame:
    """What is missing from the intended seeds x folds grid, one row per hole.

    Empty output means every (model, *group_keys) has all of `seeds` at `n_folds`.
    """
    gk = list(group_keys)
    rows = []
    for keys, d in per_run.groupby(['model'] + gk):
        keys = keys if isinstance(keys, tuple) else (keys,)
        rec = dict(zip(['model'] + gk, keys))
        for s in seeds:
            got = sorted(d.loc[d['seed'] == s, 'fold'].unique())
            if len(got) != n_folds:
                rows.append({**rec, 'seed': s, 'n_folds': len(got),
                             'folds': ','.join(map(str, got)) or '-'})
    return pd.DataFrame(rows, columns=['model'] + gk + ['seed', 'n_folds', 'folds'])


# ------------------------------------------------------------------ allele level
def allele_table(per_run: pd.DataFrame, template: str = '../../figures/template.csv',
                 group_from: str = 'full_name', label_col: str = 'HLA_Name',
                 min_count: int = 5, out: str | None = None) -> pd.DataFrame:
    """One ROC-AUC per (representation, allele), reduced folds-then-seeds.

    For every (seed, fold) run the per-allele AUC is computed on that run's
    predictions; those are averaged over the seed's folds, and the seed means are
    averaged again. The result is a single number per allele per representation,
    which is the unit the box plots and the paired tests operate on.

    Models are named and ordered by `template`, which also carries the `plot` flag
    deciding what reaches a figure. Alleles with fewer than `min_count` positives
    or negatives are dropped, matching what the notebooks did.
    """
    tmpl = pd.read_csv(template)
    tmpl = tmpl[tmpl['model'].isin(per_run['model'].unique())]
    if 'plot' in tmpl:
        tmpl = tmpl[tmpl['plot'].astype('boolean').fillna(False)]

    blocks = []
    for t in tmpl.itertuples(index=False):
        dm = per_run[per_run['model'] == t.model]
        if dm.empty:
            continue
        if dm.duplicated(['seed', 'fold']).any():
            raise ValueError(
                f"{t.model}: several rows share a (seed, fold), so a prediction "
                f"would be assigned more than once. Reduce to one row per "
                f"(model, seed, fold) first - e.g. drop the serotype split.")
        test = pd.read_csv(dm['test_path'].iloc[0])
        cols = {}
        for r in dm.itertuples(index=False):
            cols[(r.seed, r.fold)] = read_pred(r.path, r.model)

        rows = []
        for mol, dmol in test.groupby(label_col, sort=False):
            pos = int((dmol['Target'] == 1).sum())
            neg = int((dmol['Target'] == 0).sum())
            if pos < min_count or neg < min_count:
                continue
            idx = dmol.index.values
            per_seed = {}
            for (seed, fold), v in cols.items():
                try:
                    a = roc_auc_score(dmol['Target'].values, v[idx])
                except ValueError:
                    continue
                per_seed.setdefault(seed, []).append(a)
            if not per_seed:
                continue
            seed_means = {s: float(np.mean(v)) for s, v in per_seed.items()}
            rows.append({
                label_col: mol, 'pos_count': pos, 'neg_count': neg,
                'roc_auc': float(np.mean(list(seed_means.values()))),
                'roc_auc_sd': float(np.std(list(seed_means.values()), ddof=1))
                          if len(seed_means) > 1 else np.nan,
                'n_seeds': len(seed_means),
                'n_folds': int(np.mean([len(v) for v in per_seed.values()])),
            })

        blk = pd.DataFrame(rows)
        blk['model'] = t.model
        blk['group'] = getattr(t, group_from)
        blocks.append(blk)

    out_df = pd.concat(blocks, ignore_index=True)
    out_df['beta'] = out_df[label_col].map(beta_of)
    out_df['HLA_Type'] = out_df[label_col].map(hla_type)
    cols = [label_col, 'beta', 'model', 'group', 'roc_auc', 'roc_auc_sd',
            'n_seeds', 'n_folds', 'pos_count', 'neg_count', 'HLA_Type']
    out_df = out_df[cols]
    if out:
        out_df.to_csv(out, index=False)
    return out_df


def hla_type(name: str) -> str | None:
    """DP / DQ / DR / H2 bucket from an HLA name."""
    for tag in ('DP', 'DQ', 'DR', 'H2'):
        if tag in name:
            return tag
    return None


# ------------------------------------------------------------------ published tools
_REF_CACHE: dict[str, pd.DataFrame] = {}


def ref_pred(model: str) -> pd.DataFrame:
    """One published tool's predictions, keyed on (Epi_Seq, HLA_Name). Cached."""
    if model not in _REF_CACHE:
        rel, _ = REF_PRED[model]
        _REF_CACHE[model] = pd.read_csv(rp.at(rel))
    return _REF_CACHE[model]


def ref_metrics(y_true, y_score) -> dict:
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    return {'roc_auc': roc_auc_score(y_true, y_score), 'pr_auc': auc(rec, prec)}


def _ref_join(model: str, test: pd.DataFrame,
              ref_col: dict | None = None) -> tuple[pd.DataFrame, str]:
    """Test rows a tool has a score for, plus the name of its %Rank column.

    An inner join on (Epi_Seq, HLA_Name) - never on index, which is the merge that
    silently truncated 47,666 predictions to 14,117 rows in the old 3_ic notebook.
    Rows are lost here only for an allele the tool has no model for, and how many
    is reported as `n` beside every number.

    `ref_col` is the analysis's {model: column} override of REF_PRED's default; see
    the note there on why 3_ic scores NetMHCIIpan on its BA head.
    """
    col = (ref_col or {}).get(model) or REF_PRED[model][1]
    j = test.merge(ref_pred(model), on=['Epi_Seq', 'HLA_Name'], how='inner')
    return j[j[col].notna()], col


def ref_level(per_run: pd.DataFrame, group_keys=(), target_for=None,
              subsets=None, ref_col=None) -> pd.DataFrame:
    """Score the published tools on exactly the test sets the representations used.

    The test paths come out of `per_run`, not out of a second list of paths kept
    beside the config. That is the whole point: the numbers this replaces were
    typed into 2_ms/config.py as `extra_rows` and had been measured on ms_ic while
    the representations tabulated next to them were scored on ms_ql, and no code
    connected the two so nothing could catch it. Grouping per_run by test_path and
    the analysis's own group keys makes it impossible for the two sides to disagree
    about which rows they were scored on.

    There is no seed and no fold: both tools are deterministic published binaries.
    `n_seeds` is left absent rather than set to 1, so a bar drawn from these rows
    gets no error whisker (draw_bar already omits it when `<y>_sd` is missing).
    """
    gk = [k for k in group_keys if k != 'sero']
    rows = []
    for vals, d in per_run.groupby(['test_path'] + gk, dropna=False, sort=False):
        vals = vals if isinstance(vals, tuple) else (vals,)
        test_path, extra = vals[0], dict(zip(gk, vals[1:]))
        r0 = next(d.itertuples(index=False))
        tcol = target_for(r0) if target_for else 'Target'
        test = pd.read_csv(test_path)
        for model in REF_PRED:
            j, col = _ref_join(model, test, ref_col)
            for sero in (subsets or [None]):
                # same two subset forms add_metrics() accepts: an HLA_Name
                # substring, or (name, fn(frame) -> mask) for the 9-mer strata
                if sero is None:
                    sub, name = j, None
                elif isinstance(sero, str):
                    sub, name = j[j['HLA_Name'].str.contains(sero)], sero
                else:
                    name, mask_fn = sero
                    sub = j[mask_fn(j)]
                if sub[tcol].nunique() < 2:
                    continue
                rec = {'model': model, **extra}
                if sero is not None:
                    rec['sero'] = name
                rec.update(ref_metrics(sub[tcol].values, -sub[col].values))
                rec['n'] = len(sub)
                rows.append(rec)
    return pd.DataFrame(rows)


def ref_allele_level(per_run: pd.DataFrame, label_col: str = 'HLA_Name',
                     min_count: int = 5, target_for=None,
                     ref_col=None) -> pd.DataFrame:
    """One ROC-AUC per (published tool, allele), shaped like allele_table()'s rows.

    `roc_auc` is a single AUC on that allele's rows, not a folds-then-seeds mean,
    because neither tool was trained here and so has no folds and no seeds - which
    is why n_seeds and n_folds stay empty. The unit is the same molecule either
    way, which is what the box plot and the paired test need.
    """
    rows = []
    for test_path, d in per_run.groupby('test_path', dropna=False, sort=False):
        r0 = next(d.itertuples(index=False))
        tcol = target_for(r0) if target_for else 'Target'
        test = pd.read_csv(test_path)
        for model in REF_PRED:
            j, col = _ref_join(model, test, ref_col)
            for mol, dm in j.groupby(label_col, sort=False):
                pos = int((dm[tcol] == 1).sum())
                neg = int((dm[tcol] == 0).sum())
                if pos < min_count or neg < min_count:
                    continue
                rows.append({
                    label_col: mol, 'model': model, 'group': model,
                    'roc_auc': roc_auc_score(dm[tcol].values, -dm[col].values),
                    'roc_auc_sd': np.nan, 'n_seeds': np.nan, 'n_folds': np.nan,
                    'pos_count': pos, 'neg_count': neg,
                })
    out = pd.DataFrame(rows)
    if len(out):
        out['beta'] = out[label_col].map(beta_of)
        out['HLA_Type'] = out[label_col].map(hla_type)
    return out


# ------------------------------------------------------------------ LOMO level
def beta_of(group: str) -> str:
    """The beta chain a LOMO group is built around.

    The pair-split groups are '<beta>_<alpha>'; DeepNeo's are the bare beta. Its
    model has no alpha chain, so the beta is the only unit the five methods share
    and the only one a paired comparison can be defined on.
    """
    return group.split('_')[0]


def lomo_table(rep_level: pd.DataFrame, out: str | None = None) -> pd.DataFrame:
    """One row per (representation, withheld molecule), with its beta chain.

    `rep_level` is aggregate()'s second return with group_keys=['group'], i.e.
    already reduced over folds and then seeds.
    """
    df = rep_level.copy()
    df['beta'] = df['group'].map(beta_of)  # noqa: F821 - defined below
    df['HLA_Type'] = df['group'].map(hla_type)
    if out:
        # named for what it holds on the way out; `group` stays the generic key inside
        df.rename(columns={'group': 'molecule'}).to_csv(out, index=False)
    return df


def collapse_to_beta(df: pd.DataFrame, value: str = 'roc_auc',
                    unit: str | None = None) -> pd.DataFrame:
    """Average the units sharing a beta chain, giving one value per beta.

    `unit` defaults to whichever of `molecule` or `group` the frame carries.

    DeepNeo does not model the alpha chain, so everywhere it is compared against the
    pair-split models the two sides are named differently and share no keys at all:
    48 bare beta chains against 58 alpha/beta pairs allele-wise, 38 against 47 in
    LOMO. Several betas carry two to four alpha pairings.

    Averaging within the beta is what makes the five methods commensurable, because
    the beta is the only unit all of them actually have. The alternative is an
    unpaired test, which would have to assume the two sets are independent samples
    of different populations - they are neither, being the same molecules under one
    split construction. Use `unit='HLA_Name'` for the allele tables, 'molecule' for
    LOMO.
    """
    if unit is None:
        unit = 'molecule' if 'molecule' in df.columns else 'group'
    d = df.copy()
    d['beta'] = d[unit].map(beta_of)
    g = d.groupby(['model', 'beta'], as_index=False)
    out = g[value].mean()
    out['n_pairs'] = g[unit].nunique()[unit].values
    out['HLA_Type'] = out['beta'].map(hla_type)
    return out


# ------------------------------------------------------------------ statistics
def paired_wilcoxon(df: pd.DataFrame, unit: str, value: str = 'roc_auc',
                    model_col: str = 'model', models=None,
                    correction: str = 'holm') -> pd.DataFrame:
    """Wilcoxon signed-rank over every model pair, matched on `unit`.

    The units are joined by name, never by row position. The old notebook did
    `wilcoxon(a.iloc[:n], b.iloc[:n])`, which pairs the k-th row of one model with
    the k-th row of another and drops the tail - correct only if both frames happen
    to be sorted identically and equally long.

    Only units present for both models of a pair are used, so n is reported per
    comparison. `correction` is 'holm', 'bh', or None, applied across all pairs.
    """
    from itertools import combinations
    from scipy.stats import wilcoxon

    wide = df.pivot_table(index=unit, columns=model_col, values=value)
    cols = [m for m in (models or wide.columns) if m in wide.columns]

    rows = []
    for a, b in combinations(cols, 2):
        both = wide[[a, b]].dropna()
        if len(both) < 3 or np.allclose(both[a], both[b]):
            rows.append({'a': a, 'b': b, 'n': len(both), 'stat': np.nan,
                         'p': np.nan, 'median_delta': np.nan})
            continue
        stat, p = wilcoxon(both[a], both[b])
        rows.append({'a': a, 'b': b, 'n': len(both), 'stat': float(stat),
                     'p': float(p), 'median_delta': float((both[a] - both[b]).median())})
    res = pd.DataFrame(rows)

    ok = res['p'].notna()
    res['p_adj'] = np.nan
    if ok.any() and correction:
        res.loc[ok, 'p_adj'] = _adjust(res.loc[ok, 'p'].values, correction)
    elif ok.any():
        res.loc[ok, 'p_adj'] = res.loc[ok, 'p']
    res['sig'] = res['p_adj'].map(stars)
    return res


def _adjust(p: np.ndarray, method: str) -> np.ndarray:
    """Holm-Bonferroni or Benjamini-Hochberg, both monotone-enforced."""
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    s = p[order]
    if method == 'holm':
        adj = np.maximum.accumulate(s * (n - np.arange(n)))
    elif method in ('bh', 'fdr'):
        adj = np.minimum.accumulate((s * n / (np.arange(n) + 1))[::-1])[::-1]
    else:
        raise ValueError(f'unknown correction {method!r}')
    out = np.empty(n)
    out[order] = np.clip(adj, 0, 1)
    return out


def stars(p) -> str:
    if pd.isna(p):
        return ''
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'


# ------------------------------------------------------------------ driver
def portable_paths(df: pd.DataFrame) -> pd.DataFrame:
    """Strip the two roots from the path columns before a table is written.

    `path`, `file_path` and `test_path` are absolute while the analysis runs, and they resolve
    against wherever this checkout and the snapshot happen to sit. Written out as-is they differ
    on every machine, so a rerun looks like it changed the results when it did not. `src_path`
    keeps the original scratch location, which is provenance and does not depend on the reader.
    """
    out = df.copy()
    roots = [r for r in (rp.RAW_ROOT, rp.DATA_ROOT, rp._REPO) if r]
    for col in ('path', 'file_path', 'test_path'):
        if col not in out.columns:
            continue
        v = out[col].astype('string')
        for root in roots:
            v = v.str.replace(root.rstrip(os.sep) + os.sep, '', regex=False)
        out[col] = v
    return out


def run(cfg: dict, write: bool = True):
    """Run one analysis end to end from its config.py definition.

    Returns (per_run, seed_level, rep_level). With write=True the CSVs the plotting
    notebooks read are refreshed in the current directory:

        per_run_results.csv               one row per (model, seed, fold, group)
        seed_level_results.csv            folds averaged
        representation_level_results.csv  seeds averaged, with SD
        coverage.csv                      holes in the seeds x folds grid, if any
        allele_level_results.csv          one value per allele   (allele_table=True)
        lomo_level_results.csv            one value per molecule (group_keys=['group'])

    With `ref=True` the two published tools are appended to the representation
    table, to the allele table, and (through it) to the LOMO table, scored on this
    analysis's own test sets. See ref_level().
    """
    if cfg.get('extra_rows') is not None:
        raise ValueError(
            f"{cfg['name']}: extra_rows is gone. It carried published numbers as "
            f"hand-typed scalars; use ref=True, which scores the tools from "
            f"0_raw/260830/ref/ on this analysis's own test set.")
    df = collect(cfg['roots'], dir_filter=cfg.get('dir_filter', 'plots*'))
    # row_filter runs before test_for, so an analysis can drop rows whose test set
    # would not even resolve (e.g. deepneo paired with the pair-named LOMO split)
    if cfg.get('row_filter'):
        df = cfg['row_filter'](df).reset_index(drop=True)
    df['test_path'] = [cfg['test_for'](r) for r in df.itertuples(index=False)]

    met = add_metrics(df, target_for=cfg.get('target_for'), subsets=cfg.get('subsets'))
    per_run = pin_one_lr(met, cfg.get('lr_filter'))

    gk = list(cfg.get('group_keys', ()))
    seed_level, rep_level = aggregate(per_run, group_keys=gk)

    # NetMHCIIpan-4.3 and MixMHC2pred-2.0, scored here on the same test sets and
    # the same rows the representations above were scored on. This used to be
    # `extra_rows`: ten pooled scalars typed into 2_ms and 7_serotype_ms, taken
    # from a file that had been lost, and measured on ms_ic while the
    # representations beside them were scored on ms_ql. ref_level() derives the
    # test set from per_run instead, so the mismatch cannot recur.
    ref = None
    if cfg.get('ref'):
        ref = ref_level(per_run, group_keys=gk, target_for=cfg.get('target_for'),
                        subsets=cfg.get('subsets'), ref_col=cfg.get('ref_col'))
        rep_level = pd.concat([rep_level, ref], ignore_index=True)

    cov = coverage(per_run, group_keys=[k for k in gk if k != 'sero'])
    if len(cov):
        print(f'warning: {len(cov)} (model, group, seed) cells are not 5-fold '
              f'complete; see coverage.csv')

    if write:
        # `sero` and `group` are generic internally -- the slice column and the collection key.
        # Name them for what this analysis actually put in them before writing.
        out_names = {'sero': cfg.get('slice_name', 'sero'), 'group': cfg.get('group_name', 'group')}
        out_names = {k: v for k, v in out_names.items() if k != v}
        rn = (lambda d: d.rename(columns=out_names)) if out_names else (lambda d: d)

        rn(portable_paths(per_run)).to_csv('per_run_results.csv', index=False)
        rn(seed_level).to_csv('seed_level_results.csv', index=False)
        rn(rep_level).to_csv('representation_level_results.csv', index=False)
        rn(cov).to_csv('coverage.csv', index=False)

        # One value per allele, reduced the same way, for the box plots and the
        # paired tests. Not emitted for the serotype breakdowns: their per_run
        # carries one row per (seed, fold, serotype), which allele_table rejects.
        if cfg.get('allele_table'):
            al = allele_table(per_run,
                              template=cfg.get('template', '../../figures/template.csv'),
                              min_count=cfg.get('min_count', 5))
            if cfg.get('ref'):
                al = pd.concat([al, ref_allele_level(
                    per_run, min_count=cfg.get('min_count', 5),
                    target_for=cfg.get('target_for'),
                    ref_col=cfg.get('ref_col'))], ignore_index=True)
            al.rename(columns={'mean_auc': 'roc_auc', 'sd_auc': 'roc_auc_sd', 'group': 'method'}).to_csv('allele_level_results.csv', index=False)
        # One value per withheld molecule, plus the beta chain that makes DeepNeo
        # comparable with the pair-split models.
        if 'group' in gk:
            lomo_table(rep_level, out='lomo_level_results.csv')
    return per_run, seed_level, rep_level


#: the learning rate each model is pinned to, instead of letting the test metric
#: pick one. Kept here so every analysis pins it the same way.
def pin_lr(avg: pd.DataFrame) -> pd.DataFrame:
    return avg[((avg['model'] == 'deepneo') & (avg['lr'] == 'e3'))
               | ((avg['model'] != 'deepneo') & (avg['lr'] == 'e5'))
               | (avg['model'] == 'blosum')]
