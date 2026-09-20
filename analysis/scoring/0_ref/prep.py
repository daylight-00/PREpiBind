"""Build the (peptide, allele) work list and one .pep file per (tool, allele).

Why a single union
------------------
NetMHCIIpan and MixMHC2pred are training-free, so a molecule's score for a
peptide does not depend on which evaluation it is being used in. Scoring the
union of every test set once and joining afterwards therefore gives exactly the
same numbers as scoring each split separately, while making it impossible for
two splits to disagree - which is the defect being repaired here: the pooled
numbers hard-coded in scoring/2_ms/config.py were measured on ms_ic while the
representations they were tabulated against were scored on ms_ql.

Both tools are invoked once per allele over that allele's whole peptide list,
not once per peptide as 250604/1_ref/run_{net,mix}.py did. Process startup
dominates these binaries, and the %Rank columns are computed against a fixed
random-peptide background rather than against the submitted set, so batching
changes the runtime by two orders of magnitude and the scores not at all.
`verify.py` checks that claim against a per-peptide run rather than assuming it.
"""
from __future__ import annotations

import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import allelemap as am  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(os.path.dirname(HERE))          # analysis/
DATA = os.path.join(os.path.dirname(PKG), 'data', 'dataset')

#: every test set a scoring analysis scores against, and the analysis using it.
#: MS means ms_ql throughout - 2_ms and 7_serotype_ms score the representations
#: and both tools on ms_ql/test.csv and nothing reported anywhere comes from
#: another MS construction. See PROVENANCE_ONLY below.
SOURCES = {
    'full':   (f'{DATA}/full/test.csv',   '1_whole, 6_serotype'),
    'ms_ql':  (f'{DATA}/ms_ql/test.csv',  '2_ms, 7_serotype_ms'),
    'ic50':   (f'{DATA}/ic50/test.csv',   '3_ic'),
    'h2_ani': (f'{PKG}/0_raw/250513/5_hum_ani/ani_full.csv', '5_h2'),
}

#: NOT an evaluation. data/dataset/ms_ic/ is a different MS construction that no
#: analysis uses: 20,782 test rows against ms_ql's 33,490, overlapping in 6,152
#: pairs, 63% positive against 39%. The retired `extra_rows` scalars in
#: 2_ms/config.py had been measured on it, which is the split mismatch this whole
#: stage exists to repair. Its rows are scored only so verify.py can re-derive
#: those retired numbers and show where they came from - that check is what
#: identified the file an earlier audit had given up on. Nothing
#: downstream joins against it: pipeline.ref_level() takes its test paths from
#: per_run, so a number can only ever be reported on a split some representation
#: was actually scored on.
PROVENANCE_ONLY = {
    'ms_ic': f'{DATA}/ms_ic/test.csv',
}
#: LOMO holds one test.csv per withheld molecule
LOMO_GLOBS = [f'{PKG}/0_raw/250513/lomo/*/test.csv',
              f'{PKG}/0_raw/250529/lomo_beta/*/test.csv']

STD_AA = set('ACDEFGHIKLMNPQRSTVWY')


def load() -> pd.DataFrame:
    """Union of (Epi_Seq, HLA_Name) over every test set, with provenance."""
    frames = []
    for name, (path, used_by) in SOURCES.items():
        d = pd.read_csv(path, usecols=['Epi_Seq', 'HLA_Name'])
        frames.append(d.assign(src=name))
        print(f'  {name:8s} {len(d):7d} rows  ({used_by})')
    for name, path in PROVENANCE_ONLY.items():
        d = pd.read_csv(path, usecols=['Epi_Seq', 'HLA_Name'])
        frames.append(d.assign(src=name))
        print(f'  {name:8s} {len(d):7d} rows  (provenance only - no analysis '
              f'reports from this split)')
    n_lomo = 0
    lomo = []
    for pat in LOMO_GLOBS:
        for p in sorted(glob.glob(pat)):
            lomo.append(pd.read_csv(p, usecols=['Epi_Seq', 'HLA_Name']))
            n_lomo += 1
    if lomo:
        d = pd.concat(lomo, ignore_index=True)
        frames.append(d.assign(src='lomo'))
        print(f'  {"lomo":8s} {len(d):7d} rows  (4_lomo, {n_lomo} molecule dirs)')
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    print('sources:')
    allrows = load()
    pairs = allrows[['Epi_Seq', 'HLA_Name']].drop_duplicates().reset_index(drop=True)
    print(f'\nunion: {len(pairs)} unique (peptide, allele) pairs, '
          f'{pairs.HLA_Name.nunique()} alleles')

    bad = pairs[~pairs['Epi_Seq'].map(lambda s: set(s) <= STD_AA)]
    if len(bad):
        print(f'\n{len(bad)} peptide(s) contain a non-standard residue and are '
              f'dropped - neither tool scores them:')
        print(bad.to_string(index=False))
        pairs = pairs.drop(bad.index).reset_index(drop=True)

    net_ok, mix_ok = am.net_supported(), am.mix_supported()
    pairs['net'] = pairs['HLA_Name'].map(am.net)
    pairs['mix'] = pairs['HLA_Name'].map(am.mix)
    pairs.to_csv(f'{HERE}/pairs.csv', index=False)

    amap = pairs[['HLA_Name', 'net', 'mix']].drop_duplicates().copy()
    amap['net_ok'] = amap['net'].isin(net_ok)
    amap['mix_ok'] = amap['mix'].isin(mix_ok)
    amap['n_pep'] = amap['HLA_Name'].map(pairs['HLA_Name'].value_counts())
    amap = amap.sort_values('n_pep', ascending=False).reset_index(drop=True)
    amap.to_csv(f'{HERE}/allele_map.csv', index=False)

    for tool in ('net', 'mix'):
        drop = amap[~amap[f'{tool}_ok']]
        if len(drop):
            print(f'\n{tool}: {len(drop)} allele(s) the tool has no model for, '
                  f'{int(drop.n_pep.sum())} pairs excluded:')
            print(drop[['HLA_Name', tool, 'n_pep']].to_string(index=False))

    # one .pep per (tool, allele); the tool reads it whole
    for tool in ('net', 'mix'):
        d = f'{HERE}/pep/{tool}'
        os.makedirs(d, exist_ok=True)
        os.makedirs(f'{HERE}/out/{tool}', exist_ok=True)
        keep = amap.loc[amap[f'{tool}_ok'], 'HLA_Name']
        tasks = []
        for hla in keep:
            mapped = amap.loc[amap['HLA_Name'] == hla, tool].iloc[0]
            peps = pairs.loc[pairs['HLA_Name'] == hla, 'Epi_Seq'].tolist()
            with open(f'{d}/{mapped}.pep', 'w') as fh:
                fh.write('\n'.join(peps) + '\n')
            tasks.append((mapped, len(peps)))
        # longest first: with a fixed pool the wall clock is the longest task,
        # so a short one must never be scheduled ahead of a long one
        tasks.sort(key=lambda t: -t[1])
        with open(f'{HERE}/tasks_{tool}.txt', 'w') as fh:
            fh.write('\n'.join(m for m, _ in tasks) + '\n')
        tot = sum(n for _, n in tasks)
        print(f'\n{tool}: {len(tasks)} tasks, {tot} peptides, '
              f'largest allele {tasks[0][0]} with {tasks[0][1]}')


if __name__ == '__main__':
    main()
