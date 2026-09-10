"""
Shared loading, aggregation and annotation for the figure notebooks.

The notebooks under figures/ each carried their own copy of "read a metrics CSV,
merge the template, filter to the plotted models, draw, annotate with a test".
That is the same duplication scoring/pipeline.py was written to remove, and it had
already produced the same class of bug: the per-notebook copies compared models with

    wilcoxon(a.iloc[:n], b.iloc[:n])

which pairs the k-th row of one model with the k-th row of another and silently
drops the tail. It is a paired test applied to whatever order the rows happened to
be in. `paired_test()` here joins on the molecule name instead.

Everything is read from scoring/*/; nothing recomputes a metric.

    import figio as fx

    fx.rep('1_whole')                  # representation level, mean +- SD over seeds
    fx.alleles('1_whole')              # one value per allele
    fx.lomo()                          # one value per withheld molecule
    fx.rep('1_whole', refs=True)       # ... keeping NetMHCIIpan / MixMHC2pred
    fx.by_beta(fx.alleles('1_whole'))  # collapsed so DeepNeo is comparable
    fx.draw_bar(ax, ...)               # mean +- SD bars
    fx.draw_box(ax, ...)               # box + strip + significance brackets
"""
from __future__ import annotations

import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd
import seaborn as sns

_HERE = os.path.dirname(os.path.abspath(__file__))
ANAL = os.path.join(_HERE, '..', 'scoring')
sys.path.insert(0, ANAL)
import pipeline as pl  # noqa: E402

TEMPLATE = os.path.join(_HERE, 'template.csv')

#: template model ids of the two published tools. pipeline.ref_level() writes them
#: into every scoring table; the loaders here drop them unless refs=True.
REF_MODELS = ('netmhcpan', 'mixmhcpred')

#: the figure palette, in template order
PALETTE = [
    (1.0, 0.6862745098039216, 0.0),
    (0.9568627450980393, 0.4117647058823529, 0.12549019607843137),
    (0.9607843137254902, 0.19607843137254902, 0.3333333333333333),
    (0.9725490196078431, 0.3411764705882353, 0.7568627450980392),
    (0.1607843137254902, 0.7411764705882353, 0.9921568627450981),
    (0.0, 0.796078431372549, 0.7490196078431373),
    (0.00392156862745098, 0.7568627450980392, 0.34901960784313724),
    (0.615686274509804, 0.792156862745098, 0.10980392156862745),
]

def colors_for(models) -> list:
    """One PALETTE hue per method, in the order given.

    Same assignment `PALETTE[:len(d)]` made, kept as a function so a legend and a
    panel cannot drift apart: `legend_handles` calls this too. The published tools
    are last in template order, so they take the two hues after DeepNeo and every
    representation keeps the colour it has in the panels that omit them.
    """
    return [PALETTE[i % len(PALETTE)] for i in range(len(list(models)))]


METRIC_NAMES = {'roc_auc': 'ROC AUC', 'pr_auc': 'PR AUC', 'f1': 'F1 Score',
                'accuracy': 'Accuracy', 'mcc': 'MCC'}


# ---------------------------------------------------------------- loading
def template(path: str = TEMPLATE, plotted_only: bool = False) -> pd.DataFrame:
    t = pd.read_csv(path)
    if plotted_only:
        t = t[t['plot'].astype('boolean').fillna(False)]
    return t


def _merge_template(df: pd.DataFrame, tmpl_path: str, plotted_only: bool) -> pd.DataFrame:
    """Attach full_name and impose the template's row order."""
    t = template(tmpl_path, plotted_only)
    out = df.merge(t[['model', 'full_name', 'fam', 'params', 'plot']],
                   on='model', how='inner')
    order = {m: i for i, m in enumerate(t['model'])}
    return (out.assign(_o=out['model'].map(order))
               .sort_values(['_o'] + [c for c in ('dir', 'serotype', 'stratum', 'method', 'molecule') if c in out])
               .drop(columns='_o').reset_index(drop=True))


def _read(analysis: str, fname: str) -> pd.DataFrame:
    p = os.path.join(ANAL, analysis, fname)
    if not os.path.exists(p):
        raise FileNotFoundError(
            f'{p} does not exist. Run scoring/{analysis}/anal_pred_*.ipynb '
            f'(or pipeline.run(CONFIG)) first.')
    return pd.read_csv(p)


def _drop_refs(df: pd.DataFrame, refs: bool) -> pd.DataFrame:
    """Remove the published-tool rows unless the caller asked for them.

    Since pipeline.ref_level() runs as part of every analysis, every table under
    scoring/ now carries NetMHCIIpan-4.3 and MixMHC2pred-2.0 rows. Most panels
    should not show them: both tools were trained on IEDB and immunopeptidomics
    data that overlaps this test set, so their numbers are an upper bound rather
    than a like-for-like result, and a panel that puts them beside the
    representations without saying so invites the wrong reading. Opting in per
    call keeps that decision where it is being made instead of in a template flag.
    """
    return df if refs else df[~df['model'].isin(REF_MODELS)].reset_index(drop=True)


def rep(analysis: str, tmpl: str = TEMPLATE, plotted_only: bool = True,
        dir: str | None = None, sero: str | None = None,
        refs: bool = False) -> pd.DataFrame:
    """Representation level: one row per model, mean and *_sd across the three seeds.

    `refs=True` keeps NetMHCIIpan-4.3 and MixMHC2pred-2.0, which pipeline.run()
    now writes into every table. They land last because that is their template
    order, and they carry no `roc_auc_sd` - both are deterministic published
    binaries with no seed to vary - so draw_bar gives them no error whisker.
    """
    df = _drop_refs(_read(analysis, 'representation_level_results.csv'), refs)
    if dir is not None:
        df = df[df['dir'] == dir]
    if sero is not None:
        # the slice column is named for what it holds: serotype in 6/7_serotype, stratum in 8_strat
        col = 'serotype' if 'serotype' in df.columns else 'stratum'
        df = df[df[col] == sero]
    return _merge_template(df, tmpl, plotted_only)


def seed_level(analysis: str, tmpl: str = TEMPLATE, plotted_only: bool = True) -> pd.DataFrame:
    """One row per (model, seed): the mean over that seed's five folds."""
    return _merge_template(_read(analysis, 'seed_level_results.csv'), tmpl, plotted_only)


def alleles(analysis: str, tmpl: str = TEMPLATE, plotted_only: bool = True,
            refs: bool = False) -> pd.DataFrame:
    """One value per (model, allele), already reduced folds-then-seeds.

    `refs=True` keeps the two published tools. Their `roc_auc` is a single AUC on
    that allele's rows rather than a folds-then-seeds mean, because neither tool
    was trained here and so has no folds and no seeds. The unit is the same
    molecule either way, which is what the box plot and the paired test need.
    """
    return _merge_template(
        _drop_refs(_read(analysis, 'allele_level_results.csv'), refs),
        tmpl, plotted_only)


def lomo(tmpl: str = TEMPLATE, plotted_only: bool = True,
         analysis: str = '4_lomo', refs: bool = False) -> pd.DataFrame:
    """One value per (model, withheld molecule).

    `refs=True` keeps the two published tools, but read the number carefully:
    nothing is withheld from a tool that was not trained here, so its LOMO value
    is just its score on that molecule's rows while the representations' values
    beside it really are held out. The two columns do not answer the same question.
    """
    return _merge_template(
        _drop_refs(_read(analysis, 'lomo_level_results.csv'), refs),
        tmpl, plotted_only)


# ---------------------------------------------------------------- units
def by_beta(df: pd.DataFrame, value: str | None = None,
            unit: str | None = None) -> pd.DataFrame:
    """Collapse alpha/beta pairs onto their beta chain, keeping full_name.

    Needed wherever DeepNeo appears: it holds out beta chains where the other
    models hold out alpha/beta pairs, so the two sides share no unit names at all
    and a paired test would otherwise have nothing to match on. See
    pipeline.collapse_to_beta.
    """
    if unit is None:
        unit = 'HLA_Name' if 'HLA_Name' in df else 'molecule'
    if value is None:
        value = 'roc_auc' if 'roc_auc' in df else 'roc_auc'
    out = pl.collapse_to_beta(df, value=value, unit=unit)
    names = df[['model', 'full_name']].drop_duplicates()
    out = out.merge(names, on='model', how='left')
    # collapse_to_beta groups, which sorts models alphabetically. Restore the
    # template order the caller came in with: it is what assigns palette colours
    # and panel order, so a box panel would otherwise disagree with the bar panel
    # beside it while using the same colours.
    order = {m: i for i, m in enumerate(df['model'].drop_duplicates())}
    return (out.assign(_o=out['model'].map(order))
               .sort_values(['_o', 'beta'])
               .drop(columns='_o').reset_index(drop=True))


# ---------------------------------------------------------------- statistics
def paired_test(df: pd.DataFrame, unit: str, value: str, model_col: str = 'full_name',
                models=None, correction: str = 'holm') -> pd.DataFrame:
    """Holm-corrected paired Wilcoxon over every model pair, matched by unit name."""
    return pl.paired_wilcoxon(df, unit=unit, value=value, model_col=model_col,
                              models=models, correction=correction)


def letters_for(res: pd.DataFrame, means: pd.Series) -> dict:
    """Compact letter display: methods sharing a letter are not distinguished.

    Twenty-one pairwise brackets do not fit in a panel. Raising the axis until
    they do puts a ROC-AUC plot on a y-axis running past 1.0, and drawing only
    some of them needs a subset rule to defend. A letter display shows every
    comparison in one row.

    Letters come from the *maximal cliques* of the "not distinguished" graph, not
    from a single greedy pass over the methods. A greedy pass gets this wrong in a
    way that is easy to miss: with BLOSUM62 not distinguished from either
    NetMHCIIpan-4.3 or MixMHC2pred-2.0 while those two differ from each other, it
    assigns BLOSUM62 and NetMHCIIpan-4.3 different letters and so claims a
    difference the test did not find. `_check_letters` asserts the property that
    matters - every non-significant pair shares a letter, and no significant pair
    does - so a wrong display fails loudly instead of being published.

    `means` must be sorted descending: it fixes both which method is compared with
    which and the letter order, so that 'a' is the top group.
    """
    from itertools import combinations
    ns = {frozenset((r.a, r.b)) for r in res.itertuples()
          if r.sig in ('', 'ns') or pd.isna(r.p_adj)}
    methods = list(means.index)

    cliques: list[list] = []
    for k in range(len(methods), 0, -1):
        for c in combinations(methods, k):
            if all(frozenset(pair) in ns for pair in combinations(c, 2)) \
                    and not any(set(c) <= set(e) for e in cliques):
                cliques.append(list(c))

    # Assign letters by rank, so 'a' is the top group and the letters read in the
    # same order as the values. Enumerating cliques by size finds them in an order
    # that has nothing to do with performance - it gave the third-best method 'e'.
    rank = {m: i for i, m in enumerate(methods)}
    cliques.sort(key=lambda g: min(rank[m] for m in g))

    from string import ascii_lowercase
    out = {m: '' for m in methods}
    for letter, group in zip(ascii_lowercase, cliques):
        for m in group:
            out[m] += letter
    _check_letters(out, res)
    return out


def _check_letters(letters: dict, res: pd.DataFrame) -> None:
    """A letter display is only readable if it means what it looks like."""
    for r in res.itertuples():
        if r.a not in letters or r.b not in letters:
            continue
        shared = set(letters[r.a]) & set(letters[r.b])
        quiet = r.sig in ('', 'ns') or pd.isna(r.p_adj)
        if quiet and not shared:
            raise AssertionError(
                f'letter display would claim {r.a} differs from {r.b}, but the '
                f'test did not distinguish them (p_adj={r.p_adj})')
        if not quiet and shared:
            raise AssertionError(
                f'letter display would claim {r.a} and {r.b} are alike, but '
                f'p_adj={r.p_adj:.2e} ({r.sig})')


# ---------------------------------------------------------------- drawing
def draw_bar(ax, data: pd.DataFrame, y: str = 'roc_auc', x: str = 'full_name',
             title: str = '', ylim=(0.4, 1.0), show_xlabel: bool = True,
             label_with_value: bool = True, errorbar: bool = True):
    """One bar per representation: the mean over seeds, with the seed SD as the error bar.

    The error bar is the SD of three seed-level values (each itself a mean over five
    folds), which is what the benchmark reports. Published baselines carry a single
    number and no SD, so they get no whisker.
    """
    d = data.copy()
    colors = colors_for(d['model']) if 'model' in d else PALETTE[:len(d)]
    ax.bar(range(len(d)), d[y], color=colors,
           yerr=(d.get(f'{y}_sd').fillna(0) if errorbar and f'{y}_sd' in d else None),
           capsize=3, error_kw={'lw': 1.0, 'ecolor': '0.25'})
    ax.set_xticks(range(len(d)))
    if show_xlabel:
        labels = ([f'{v:.3f}' for v in d[y]] if label_with_value
                  else d[x].tolist())
        ax.set_xticklabels(labels, rotation=90)
    else:
        ax.set_xticklabels([])
    ax.set_title(title)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_ylim(ylim)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    return ax


def common_units(data: pd.DataFrame, unit: str, x: str = 'full_name',
                 among=None) -> pd.DataFrame:
    """Keep only the units every method has a value for, or every method in `among`.

    Needed once NetMHCIIpan is in a panel: it has no model for H2-IAg7, so it
    carries one allele fewer than everything else. paired_wilcoxon joins each pair
    separately, so leaving that alone gives one Holm family whose n differs
    between pairs - the corrected p values would be adjusted across tests run on
    different samples. Dropping the one unit costs almost nothing and makes every
    comparison the same size.

    `among` names the methods that must agree, for a panel holding a method whose
    units are named differently on purpose. DeepNeo is that method when the unit
    is the un-collapsed `HLA_Name`: it models no alpha chain, so its alleles are
    bare beta chains and it shares no name at all with the pair-named rest. It
    still belongs in the panel as a plotted distribution; it simply cannot be
    paired with anything, and requiring it to agree would empty the frame.
    """
    d = data if among is None else data[data[x].isin(among)]
    n = d[x].nunique()
    keep = d.groupby(unit)[x].nunique()
    keep = keep[keep == n].index
    lost = sorted(set(d[unit]) - set(keep))
    if lost:
        print(f'  common_units: dropped {len(lost)} unit(s) not shared by all '
              f'{n} of {"" if among is None else "the paired "}methods: '
              f'{", ".join(map(str, lost))}')
    ok = data[unit].isin(keep)
    if among is not None:
        # a method outside `among` keeps all of its rows: its units are named on a
        # different scheme, so `keep` says nothing about them and testing them
        # against it would drop the method from the panel entirely
        ok |= ~data[x].isin(among)
    return data[ok].reset_index(drop=True)


def draw_box(ax, data: pd.DataFrame, unit: str, y: str = 'roc_auc',
             x: str = 'full_name', title: str = '', ylim=(0.4, 1.0),
             show_xlabel: bool = True, annotate: bool = True,
             correction: str = 'holm', bracket_y_base=None, bracket_height=None,
             label_with_value: bool = True, verbose: bool = False,
             shared_units=False,
             stat_models=None, letter_display: bool = False,
             bracket_only=None):
    """Box + strip over the per-unit values, with paired-Wilcoxon brackets.

    `data` is long: one row per (model, unit). `unit` names the column holding the
    allele or withheld molecule - the statistical unit. Every model must already be
    reduced to one value per unit, which is what the scoring tables provide; passing
    per-fold or per-seed rows here would pseudo-replicate.

    Brackets show Holm-corrected p across all pairs tested in this panel. Only
    significant pairs get a bracket, as before. `stat_models` restricts which
    methods are tested at all. `letter_display` replaces the brackets with one row
    of letters, which is what a panel holding more than about five methods needs -
    see letters_for().

    `bracket_only` draws only the pairs involving the named methods, without
    touching the correction - for a panel whose significant pairs outnumber the
    bracket rows that physically fit. `bracket_upper_bound`, which additionally
    filtered by direction, is gone: the direction caveat belongs in the caption,
    not in a silent drawing rule.
    """
    d = data.copy()
    if shared_units is not False and shared_units is not None:
        d = common_units(d, unit, x,
                         among=None if shared_units is True else shared_units)
    order = [g for g in d[x].drop_duplicates()]
    label_map = {g: f'{d.loc[d[x] == g, y].mean():.3f}' for g in order}
    d['_x'] = pd.Categorical(d[x], categories=order, ordered=True)
    if 'model' in d:
        ids = d.drop_duplicates(x).set_index(x)['model'].reindex(order)
        palette = colors_for(ids)
    else:
        palette = PALETTE[:len(order)]

    sns.boxplot(data=d, x='_x', y=y, hue='_x', order=order, palette=palette,
                showfliers=False, ax=ax, legend=False)
    # stripplot's jitter draws from the global numpy RNG, so without this the dots land somewhere
    # new on every run and the figure never reproduces. The offsets are cosmetic; the seed only
    # fixes them.
    np.random.seed(0)
    sns.stripplot(data=d, x='_x', y=y, order=order, color='black', size=3,
                  jitter=True, alpha=0.5, ax=ax)

    # A two-sided Wilcoxon signed-rank test on n pairs has a smallest attainable
    # p of 2 / 2**n, so below n = 6 (p >= 0.0625) no result can ever be starred and
    # brackets would only suggest a test was capable of finding something. The H2
    # panels sit here: 3 withheld H2 molecules, 8 H2 alleles. They stay descriptive.
    n_units = d[unit].nunique() if stat_models is None else \
        d.loc[d[x].isin(stat_models), unit].nunique()
    if annotate and n_units < 6:
        print(f'{title or "panel"}: n={n_units} units, below the n=6 floor where a '
              f'two-sided signed-rank test can reach p<0.05. Left unannotated.')
        annotate = False

    # `stat_models` limits which methods are tested at all, not just which
    # brackets are drawn. Use it for a method that is in the panel to be looked at
    # but not to be tested against: NetMHCIIpan-4.3 and MixMHC2pred-2.0 were
    # trained on data overlapping this test set, so a p value comparing a
    # representation to them is not a p value about relative accuracy. Leaving
    # them out of the family also means they do not inflate the Holm correction
    # applied to the comparisons that are meaningful. It is not used by any panel
    # now: the Per Molecule panel tests every method it shows.
    tested = [g for g in order if stat_models is None or g in stat_models]
    res = None
    if annotate and len(tested) > 1:
        res = paired_test(d[d[x].isin(tested)], unit=unit, value=y, model_col=x,
                          models=tested, correction=correction)
        if verbose:
            print(f'--- {title} ---')
            print(res[['a', 'b', 'n', 'median_delta', 'p', 'p_adj', 'sig']].to_string(index=False))
        if letter_display:
            means = d[d[x].isin(tested)].groupby(x)[y].mean().sort_values(ascending=False)
            letters = letters_for(res, means)
            top = ylim[1] - (ylim[1] - ylim[0]) * 0.06
            for i, g in enumerate(order):
                if g in letters:
                    ax.text(i, top, letters[g], ha='center', va='bottom',
                            fontsize=10)
            annotate = False

    if annotate and len(tested) > 1:
        pos = {g: i for i, g in enumerate(order)}
        if bracket_y_base is None:
            bracket_y_base = ylim[1] - (ylim[1] - ylim[0]) * 0.05
        if bracket_height is None:
            bracket_height = (ylim[1] - ylim[0]) * 0.04

        sig = {(r.a, r.b): r.sig for r in res.itertuples() if r.sig not in ('', 'ns')}
        if bracket_only:
            # Draw only pairs involving one of these methods. This is a *drawing*
            # filter: every pair stays in the Holm family, because that is the
            # multiple-comparison burden actually incurred, and every p value is
            # reported in the supplementary. It exists because the panel has a hard
            # capacity - measured from the data, about six bracket rows fit between
            # the highest whisker and the top of the axis - and the significant
            # pairs exceed it. Raising the axis until they fit would put a ROC-AUC
            # plot well past 1.0 and break the shared scale with the panel beside it.
            want = set(bracket_only)
            sig = {k: v for k, v in sig.items() if want & set(k)}
        pairs = sorted(combinations(order, 2),
                       key=lambda ab: (-abs(pos[ab[0]] - pos[ab[1]]), pos[ab[0]], pos[ab[1]]))
        far = [ab for ab in pairs if abs(pos[ab[0]] - pos[ab[1]]) > 1]
        near = [ab for ab in pairs if abs(pos[ab[0]] - pos[ab[1]]) == 1]

        count = 0
        for a, b in far:
            s = sig.get((a, b)) or sig.get((b, a))
            if not s:
                continue
            _bracket(ax, pos[a], pos[b], bracket_y_base - count * bracket_height,
                     bracket_height, s)
            count += 1
        near_y = bracket_y_base - count * bracket_height
        for a, b in near:
            s = sig.get((a, b)) or sig.get((b, a))
            if not s:
                continue
            _bracket(ax, pos[a], pos[b], near_y, bracket_height, s)

    ax.set_xticks(range(len(order)))
    if show_xlabel:
        ax.set_xticklabels([label_map[g] if label_with_value else g for g in order],
                           rotation=90)
    else:
        ax.set_xticklabels([])
    ax.set_title(title)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_ylim(ylim)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    return ax


def _bracket(ax, x1, x2, y, h, label):
    ax.plot([x1, x1, x2, x2], [y, y + h * 0.3, y + h * 0.3, y], lw=1.0, c='k')
    ax.text((x1 + x2) / 2, y - h * 0.1, label, ha='center', va='bottom')


def legend_handles(names, models=None):
    """Figure-level legend swatches, matching what draw_bar / draw_box painted.

    Pass `models` (the template ids, same order as `names`) whenever the panel
    includes a published tool, so the swatch is the grey the panel actually used
    rather than the next palette hue.
    """
    import matplotlib.pyplot as plt
    colors = colors_for(models) if models is not None else PALETTE[:len(names)]
    return ([plt.Rectangle((0, 0), 1, 1, facecolor=c) for c in colors], list(names))
