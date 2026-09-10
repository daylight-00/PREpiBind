"""Merge out/<tool>/<allele>.csv into one prediction table per tool.

Output (next to this file):

    pred_netmhcpan.csv   Epi_Seq, HLA_Name, el_score, el_rank, ba_score, ba_rank, ba
    pred_mixmhcpred.csv  Epi_Seq, HLA_Name, rank
    coverage.csv         per (tool, allele): peptides asked for, scored, missing

These are keyed on (Epi_Seq, HLA_Name) and cover the union of every test set, so
any analysis downstream is an inner join - no split can be scored against a
number measured on a different split.

Which column is the score
-------------------------
`el_rank` and `rank` are the two tools' %Rank against a fixed random-peptide
background, lower meaning better binder, and they are the only *comparable*
quantity the two tools share. MixMHC2pred-2.0 does emit raw scores under its -e
flag (Score_, ScorePWM_), which score.py does not pass; none is an affinity, its
README warns Score_ cannot be transformed to the %Rank, and ScorePWM_ is
unbounded and PWM-only, so neither is on a shared scale with Score_EL. So a
comparison between the tools has to run on -%Rank, and `metrics.py` uses that.
See docs/notes/2026-09-01-mixmhc2pred-no-affinity-head.md.

`el_score` is NetMHCIIpan's raw 0-1 output. It is kept because %Rank is
normalised *within an allele*: pooling %Rank over 126 alleles is not the same
ordering as pooling el_score, which is why the two give different pooled AUCs
(0.9718 vs 0.9693 on the old ms_ic rows). Within one allele the two are
monotone in each other up to %Rank's 2-decimal rounding, so per-allele AUCs
agree; metrics.py reports both so that is visible rather than assumed.
"""
from __future__ import annotations

import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = {'net': 'pred_netmhcpan.csv', 'mix': 'pred_mixmhcpred.csv'}


def main() -> None:
    pairs = pd.read_csv(f'{HERE}/pairs.csv')
    amap = pd.read_csv(f'{HERE}/allele_map.csv')
    cov = []
    for tool, dst in TOOLS.items():
        rows, ok = [], amap[amap[f'{tool}_ok']]
        for t in ok.itertuples(index=False):
            mapped = getattr(t, tool)
            p = f'{HERE}/out/{tool}/{mapped}.csv'
            asked = int(t.n_pep)
            if not os.path.exists(p):
                cov.append({'tool': tool, 'HLA_Name': t.HLA_Name, 'mapped': mapped,
                            'asked': asked, 'scored': 0, 'missing': asked})
                continue
            d = pd.read_csv(p)
            d['HLA_Name'] = t.HLA_Name
            val = [c for c in d.columns if c not in ('Epi_Seq', 'HLA_Name')]
            scored = int(d[val[0]].notna().sum())
            cov.append({'tool': tool, 'HLA_Name': t.HLA_Name, 'mapped': mapped,
                        'asked': asked, 'scored': scored, 'missing': asked - scored})
            rows.append(d)
        if not rows:
            print(f'{tool}: nothing collected')
            continue
        out = pd.concat(rows, ignore_index=True)
        val = [c for c in out.columns if c not in ('Epi_Seq', 'HLA_Name')]
        out = out[['Epi_Seq', 'HLA_Name'] + val]
        out.to_csv(f'{HERE}/{dst}', index=False)
        print(f'{tool}: {len(out)} rows, {out.HLA_Name.nunique()} alleles -> {dst}')

    c = pd.DataFrame(cov)
    c.to_csv(f'{HERE}/coverage.csv', index=False)
    for tool in TOOLS:
        t = c[c.tool == tool]
        gap = t[t.missing > 0]
        print(f'{tool}: asked {int(t.asked.sum())}, scored {int(t.scored.sum())}, '
              f'{len(gap)} allele(s) incomplete')
        if len(gap):
            print(gap[['HLA_Name', 'mapped', 'asked', 'scored', 'missing']]
                  .to_string(index=False))
    print(f'\nunion asked for: {len(pairs)} pairs '
          f'(per-tool totals differ by the unsupported alleles)')


if __name__ == '__main__':
    main()
