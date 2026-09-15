"""Supplementary tables for the NetMHCIIpan-4.3 / MixMHC2pred-2.0 comparison.

Writes, next to this file:

    supp_ref_tables.tex   three booktabs tables in the manuscript's house style
    supp_ref_pooled.csv   pooled ROC-AUC / PR-AUC, every method x every analysis
    supp_ref_molecule.csv per-molecule summary, every method x every panel
    supp_ref_pairs.csv    all 21 Holm-corrected paired Wilcoxon tests for Per Molecule
    supp_ref_pairs_beta.csv  the same 21 pairs on the beta chain, incl. DeepNeo
    supp_ref_pairs_lomo.csv  the 10 LOMO pairs on the beta chain (Figure 3b)

The published tools were trained on data that overlap this benchmark. Their scores
are therefore retained as contextual references rather than treated as independent
external validation. Detailed scoring conventions are described in Supplementary
Text 1.8; the table captions below are kept concise but retain the local context
needed to interpret each table without repeating the full Methods.

Everything is read from scoring/*/; nothing is recomputed here.
"""
from __future__ import annotations

import os

import pandas as pd

import figio as fx

HERE = os.path.dirname(os.path.abspath(__file__))

#: (column label, analysis, slice column, slice value)
POOLED = [
    ('Qualitative',   '1_whole', None,  None),
    ('MS',            '2_ms',    'dir', 'plots_ql'),
    ('IC50 $<$500',   '3_ic',    'dir', 'plots_500'),
    ('IC50 $<$1000',  '3_ic',    'dir', 'plots_1000'),
    ('H2-Out',        '5_h2',    None,  None),
]


def pooled() -> pd.DataFrame:
    out = {}
    for label, analysis, gcol, gval in POOLED:
        kw = {gcol: gval} if gcol else {}
        d = fx.rep(analysis, refs=True, **kw)
        if gcol is None:
            for c in ('dir', 'serotype', 'stratum'):
                if c in d:
                    d = d[d[c].isna()]
        out[label] = d.set_index('full_name')['roc_auc']
        if 'n' in d:
            out[f'{label} $n$'] = d.set_index('full_name')['n']
    return pd.DataFrame(out)


def conventions() -> pd.DataFrame:
    """Qualitative per-allele ROC-AUC under three unit conventions."""
    a = fx.alleles('1_whole', refs=True)
    paired = [m for m in a['full_name'].drop_duplicates() if m != 'DeepNeo']
    sub = fx.common_units(a, 'HLA_Name', among=paired)
    b = fx.by_beta(a)
    out = pd.DataFrame({
        'all': a.groupby('full_name', sort=False)['roc_auc'].mean(),
        '$n$': a.groupby('full_name', sort=False)['roc_auc'].size(),
        'paired': sub.groupby('full_name', sort=False)['roc_auc'].mean(),
        'beta': b.groupby('full_name', sort=False)['roc_auc'].mean(),
        '$n$ (beta)': b.groupby('full_name', sort=False)['roc_auc'].size(),
    })
    out['all $-$ beta'] = out['all'] - out['beta']
    return out


def molecule() -> pd.DataFrame:
    _a = fx.alleles('1_whole', refs=True)
    _p = [m for m in _a['full_name'].drop_duplicates() if m != 'DeepNeo']
    whole = fx.common_units(_a, 'HLA_Name', among=_p)
    lomo = fx.by_beta(fx.lomo(refs=True))
    h2 = fx.by_beta(fx.alleles('5_h2', refs=True), value='roc_auc', unit='HLA_Name')
    panels = {
        'Per Molecule': (whole, 'roc_auc'),
        'LOMO -- Whole': (lomo, 'roc_auc'),
        'LOMO -- H2': (lomo[lomo['beta'].str.contains('H2')], 'roc_auc'),
        'H2-Out': (h2, 'roc_auc'),
    }
    out = {}
    for label, (d, y) in panels.items():
        g = d.groupby('full_name', sort=False)[y]
        out[label] = g.mean()
        out[f'{label} n'] = g.size()
    return pd.DataFrame(out)


def pairs() -> pd.DataFrame:
    """Paired tests for the allele-unit Per Molecule panel."""
    a = fx.alleles('1_whole', refs=True)
    pairable = [m for m in a['full_name'].drop_duplicates() if m != 'DeepNeo']
    d = fx.common_units(a, 'HLA_Name', among=pairable)
    d = d[d['full_name'].isin(pairable)]
    res = fx.paired_test(d, unit='HLA_Name', value='roc_auc', models=pairable)
    return res[['a', 'b', 'n', 'median_delta', 'p', 'p_adj', 'sig']]


def pairs_beta() -> pd.DataFrame:
    """Paired tests after collapsing all methods onto the beta-chain unit."""
    b = fx.common_units(fx.by_beta(fx.alleles('1_whole', refs=True)), 'beta')
    order = list(b['full_name'].drop_duplicates())
    res = fx.paired_test(b, unit='beta', value='roc_auc', models=order)
    return res[['a', 'b', 'n', 'median_delta', 'p', 'p_adj', 'sig']]


def pairs_lomo() -> pd.DataFrame:
    """Paired tests behind the LOMO panel, at the beta-chain unit.

    The published tools are excluded rather than kept as references: nothing was
    withheld from them, so they have no leave-one-molecule-out value to pair.
    """
    b = fx.by_beta(fx.lomo())
    order = list(b['full_name'].drop_duplicates())
    res = fx.paired_test(b, unit='beta', value='roc_auc', models=order)
    return res[['a', 'b', 'n', 'median_delta', 'p', 'p_adj', 'sig']]


def _tex(df: pd.DataFrame, caption: str, label: str, fmt: dict) -> str:
    cols = 'l' + 'r' * len(df.columns)
    head = ' & '.join([''] + [f'\\textbf{{{c}}}' for c in df.columns]) + ' \\\\'
    lines = []
    for name, row in df.iterrows():
        cells = [fmt.get(c, '{:.3f}').format(row[c]) if pd.notna(row[c]) else '--'
                 for c in df.columns]
        bold = name in ('NetMHCIIpan-4.3', 'MixMHC2pred-2.0')
        nm = f'\\textit{{{name}}}' if bold else name
        lines.append(' & '.join([nm] + cells) + ' \\\\')
    return ('\\begin{table}[H]\n\\centering\n\\small\n'
            f'\\caption{{{caption}}}\n\\label{{{label}}}\n'
            f'\\begin{{tabular}}{{{cols}}}\n\\toprule\n{head}\n\\midrule\n'
            + '\n'.join(lines)
            + '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n')


def main() -> None:
    p, c, m, w, wb = pooled(), conventions(), molecule(), pairs(), pairs_beta()
    wl = pairs_lomo()
    p.to_csv(f'{HERE}/supp_ref_pooled.csv')
    c.to_csv(f'{HERE}/supp_ref_convention.csv')
    m.to_csv(f'{HERE}/supp_ref_molecule.csv')
    w.to_csv(f'{HERE}/supp_ref_pairs.csv', index=False)
    wb.to_csv(f'{HERE}/supp_ref_pairs_beta.csv', index=False)
    wl.to_csv(f'{HERE}/supp_ref_pairs_lomo.csv', index=False)

    tex = [
        _tex(
            p,
            'Pooled ROC-AUC and effective sample sizes for all representations and the two '
            'published tools. The tools are scored only on supported alleles and were not '
            'retrained on our splits, so their values are contextual rather than like-for-like. '
            'NetMHCIIpan-4.3 uses its EL rank outside IC50 and BA rank for IC50; '
            'MixMHC2pred-2.0 uses its percentile-rank output.',
            'tab:supp-ref-pooled',
            {col: '{:,.0f}' for col in p.columns if col.endswith('$n$')},
        ),
        _tex(
            c,
            'Qualitative per-allele ROC-AUC under three unit conventions. \\textbf{all} uses '
            'every allele supported by each method; \\textbf{paired} uses the 57 '
            '$\\alpha$/$\\beta$ alleles shared by the pair-named methods and is used in '
            'Figure~3a; \\textbf{beta} collapses to the $\\beta$ chain so DeepNeo can be '
            'paired with the others. Differences between conventions are small and do not '
            'support an ordering between ESMC 300M and ESM3 Small.',
            'tab:supp-ref-convention',
            {'$n$': '{:.0f}', '$n$ (beta)': '{:.0f}', 'all $-$ beta': '{:+.3f}'},
        ),
        _tex(
            m[[col for col in m.columns if not col.endswith(' n')]],
            'Mean per-molecule ROC-AUC. The Per Molecule column uses the allele unit of '
            'Figure~3a; LOMO and H2-out use the $\\beta$-chain unit. Published tools were '
            'not retrained under the leave-out splits and are shown only as contextual '
            'references. NetMHCIIpan-4.3 lacks H2-IAg7 coverage and therefore contributes '
            'seven H2-out molecules. Counts are given in '
            'Table~\\ref{tab:supp-ref-molecule-n}.',
            'tab:supp-ref-molecule',
            {},
        ),
        _tex(
            m[[col for col in m.columns if col.endswith(' n')]].rename(
                columns=lambda col: col[:-2]
            ),
            'Number of molecules behind each mean in '
            'Table~\\ref{tab:supp-ref-molecule}.',
            'tab:supp-ref-molecule-n',
            {col: '{:.0f}' for col in m.columns},
        ),
        _tex(
            w.set_index(w['a'] + ' vs ' + w['b']).drop(columns=['a', 'b']).rename(
                columns={
                    'n': '$n$',
                    'median_delta': '$\\Delta$',
                    'p': '$p$',
                    'p_adj': '$p_{\\mathrm{adj}}$',
                    'sig': 'Sig.',
                }
            ),
            'Holm-corrected paired Wilcoxon tests for the 57 shared allele pairs in '
            'Figure~3a. $\\Delta$ is the median paired difference ($a-b$). DeepNeo is '
            'excluded at this unit because it models only the $\\beta$ chain; its '
            'comparisons are reported in Table~\\ref{tab:supp-ref-pairs-beta}. '
            'Published-tool comparisons are included for completeness but should be '
            'interpreted in light of their overlapping training data.',
            'tab:supp-ref-pairs',
            {
                '$n$': '{:.0f}',
                '$\\Delta$': '{:+.3f}',
                '$p$': '{:.2e}',
                '$p_{\\mathrm{adj}}$': '{:.2e}',
                'Sig.': '{}',
            },
        ),
        _tex(
            wb.set_index(wb['a'] + ' vs ' + wb['b']).drop(columns=['a', 'b']).rename(
                columns={
                    'n': '$n$',
                    'median_delta': '$\\Delta$',
                    'p': '$p$',
                    'p_adj': '$p_{\\mathrm{adj}}$',
                    'sig': 'Sig.',
                }
            ),
            'The same paired comparison after collapsing all methods onto the '
            '$\\beta$-chain unit ($n=47$), which permits direct comparison with DeepNeo. '
            '$\\Delta$ is the median paired difference ($a-b$). Published-tool comparisons '
            'remain contextual because those tools were not trained on the present splits.',
            'tab:supp-ref-pairs-beta',
            {
                '$n$': '{:.0f}',
                '$\\Delta$': '{:+.3f}',
                '$p$': '{:.2e}',
                '$p_{\\mathrm{adj}}$': '{:.2e}',
                'Sig.': '{}',
            },
        ),
    ]
    with open(f'{HERE}/supp_ref_tables.tex', 'w') as fh:
        fh.write('% generated by figures/supp_ref_tables.py - do not hand-edit\n'
                 '% see docs/notes/2026-08-30-ref-rescore.md\n\n' + '\n'.join(tex))

    print(p.round(3).to_string())
    print()
    print(c.round(4).to_string())
    print()
    print(m.round(3).to_string())
    print(f'\nwrote supp_ref_tables.tex and four CSVs '
          f'({len(w)} pairs, {w.sig.ne("ns").sum()} significant)')


if __name__ == '__main__':
    main()
