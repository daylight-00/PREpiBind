"""Three checks on the rescored predictions. Run after run.sh.

1. Batching is a no-op. Both tools are invoked once per allele here; 250604's
   scripts invoked them once per peptide. If a %Rank were computed against the
   submitted set rather than a fixed background, batching would change every
   number. A sample of peptides is rescored one at a time and compared.

2. The retired `extra_rows` scalars are reproduced. MS means ms_ql throughout; scoring/2_ms/config.py carries
   roc_auc=0.9718296579610518 for netmhcpan and 0.950912204811488 for mixmhcpred
   as hand-entered `extra_rows`, and docs/notes/2026-08-30-value-update.md records that the file
   behind them could not be identified. They had been measured on the unused
   ms_ic construction rather than on the MS split, which is the mismatch this
   stage repairs. Scoring the new predictions on those rows should land on the
   retired values - that is what makes them traceable, and it is the only reason
   ms_ic appears anywhere in this pipeline (prep.PROVENANCE_ONLY). No reported
   number comes from it.

3. Nothing is missing. Every (peptide, allele) asked for has a score, except the
   alleles the tool has no model for, which are listed explicitly.
"""
from __future__ import annotations

import os
import random
import subprocess
import sys

import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', '..'))
import rawpath as rp  # noqa: E402
import score as sc  # noqa: E402

N_SAMPLE = 40
#: the retired `extra_rows` scalars, each with the split it was actually measured
#: on. Only NetMHCIIpan's came from the unused ms_ic construction; MixMHC2pred's
#: were on the MS split all along and reproduce to fifteen decimal places once the
#: allele-coverage convention is applied. That asymmetry is the reason a hand-typed
#: scalar cannot be trusted: two cells side by side in one table, one right and one
#: wrong, with nothing in the code to tell them apart.
OLD = {'net': (0.9718296579610518, 'ms_ic'),
       'mix': (0.950912204811488,  'ms_ql')}
DATA = os.path.dirname(rp.data('dataset/full/test.csv'))[:-len('/full')]

#: the collected predictions, read from the snapshot rather than the scratch run
PRED = {'net': rp.at('260830/ref/pred_netmhcpan.csv'),
        'mix': rp.at('260830/ref/pred_mixmhcpred.csv')}


def check_batching(rng: random.Random) -> bool:
    print('1. batched vs per-peptide')
    ok = True
    for tool, col in (('net', 'el_rank'), ('mix', 'rank')):
        pred = pd.read_csv(PRED[tool])
        amap = pd.read_csv(f'{HERE}/allele_map.csv')
        m = dict(zip(amap['HLA_Name'], amap[tool]))
        sample = pred.sample(N_SAMPLE, random_state=0)
        bad = 0
        for r in sample.itertuples(index=False):
            mapped = m[r.HLA_Name]
            pep = os.path.join(HERE, 'one.pep')
            with open(pep, 'w') as fh:
                fh.write(r.Epi_Seq + '\n')
            import tempfile
            with tempfile.TemporaryDirectory() as tmp:
                got = (sc.run_net if tool == 'net' else sc.run_mix)(pep, mapped, tmp)
            alone = float(got[r.Epi_Seq]['el_rank' if tool == 'net' else 'rank'])
            batched = float(getattr(r, col))
            if abs(alone - batched) > 1e-9:
                bad += 1
                if bad <= 3:
                    print(f'   MISMATCH {tool} {r.Epi_Seq} {r.HLA_Name}: '
                          f'alone={alone} batched={batched}')
        os.path.exists(f'{HERE}/one.pep') and os.remove(f'{HERE}/one.pep')
        print(f'   {tool}: {N_SAMPLE - bad}/{N_SAMPLE} identical')
        ok &= bad == 0
    return ok


def check_old_numbers() -> bool:
    print('\n2. reproduce the retired extra_rows scalars (provenance only)')
    ok = True
    for tool, col in (('net', 'el_rank'), ('mix', 'rank')):
        old, split = OLD[tool]
        test = pd.read_csv(f'{DATA}/{split}/test.csv')
        pred = pd.read_csv(PRED[tool])
        d = test.merge(pred, on=['Epi_Seq', 'HLA_Name'], how='inner').dropna(subset=[col])
        got = roc_auc_score(d['Target'], -d[col])
        delta = got - old
        flag = 'EXACT' if abs(delta) < 1e-12 else f'delta={delta:+.2e}'
        print(f'   {tool}: on {split}, n={len(d):6d}  roc_auc={got:.15f}  '
              f'retired={old:.15f}  {flag}')
        ok &= abs(delta) < 1e-3
    return ok


def check_coverage() -> bool:
    print('\n3. coverage')
    c = pd.read_csv(f'{HERE}/coverage.csv')
    ok = True
    for tool in ('net', 'mix'):
        t = c[c.tool == tool]
        gap = t[t.missing > 0]
        print(f'   {tool}: {int(t.scored.sum())}/{int(t.asked.sum())} pairs scored, '
              f'{len(gap)} allele(s) short')
        if len(gap):
            print(gap[['HLA_Name', 'mapped', 'asked', 'scored', 'missing']]
                  .to_string(index=False))
        ok &= gap.empty
    return ok


if __name__ == '__main__':
    rng = random.Random(0)
    results = [check_batching(rng), check_old_numbers(), check_coverage()]
    print('\n' + ('all checks passed' if all(results)
                  else f'{results.count(False)} check(s) failed'))
    sys.exit(0 if all(results) else 1)
