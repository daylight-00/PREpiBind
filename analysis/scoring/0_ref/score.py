#!/usr/bin/env python
"""Score one allele's whole peptide list with one external tool.

    score.py net DRB1_0701
    score.py mix DRB1_07_01

Writes out/<tool>/<mapped>.csv and is idempotent: an existing output whose row
count already matches the .pep is left alone, so a partially finished job is
resumed by rerunning the same task list.

Parsing
-------
Both tables are read by *column name from the left*, never by counting back from
the end of the line. NetMHCIIpan appends a `BindLevel` field ('<= SB' / '<= WB')
only to rows above a threshold, so the number of trailing fields varies per row;
250604/1_ref/run_net.py indexed from the right (`data[-6]`) after splitting the
line on '<', which happens to work but breaks silently the moment a column is
added. Rows are joined back to peptides by the Peptide column, because
NetMHCIIpan returns its table sorted alphabetically, not in input order.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
NET = os.path.join(os.environ.get('NETMHCIIPAN_HOME', ''), 'netMHCIIpan')
MIX = os.path.join(os.environ.get('MIXMHC2PRED_HOME', ''), 'MixMHC2pred_unix')

#: NetMHCIIpan-4.3 table, leading fields in order
NET_COLS = ['Pos', 'MHC', 'Peptide', 'Of', 'Core', 'Core_Rel', 'Inverted',
            'Identity', 'el_score', 'el_rank', 'Exp_Bind', 'ba_score',
            'ba_rank', 'ba']
NET_KEEP = ['el_score', 'el_rank', 'ba_score', 'ba_rank', 'ba']


def run_net(pep: str, allele: str, tmp: str) -> dict[str, dict]:
    env = dict(os.environ, TMPDIR=tmp)
    r = subprocess.run([NET, '-a', allele, '-inptype', '1', '-BA', '-f', pep],
                       capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise RuntimeError(f'netMHCIIpan exit {r.returncode}: {r.stderr[-2000:]}')
    out = {}
    for ln in r.stdout.splitlines():
        f = ln.split()
        if len(f) < len(NET_COLS) or not f[0].isdigit():
            continue
        row = dict(zip(NET_COLS, f))
        out[row['Peptide']] = {k: row[k] for k in NET_KEEP}
    if not out:
        raise RuntimeError(f'netMHCIIpan produced no rows for {allele}\n'
                           f'{r.stdout[-2000:]}')
    return out


def run_mix(pep: str, allele: str, tmp: str) -> dict[str, dict]:
    dst = os.path.join(tmp, 'mix.out')
    r = subprocess.run([MIX, '-i', pep, '-a', allele, '--no_context', '-o', dst],
                       capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(dst):
        raise RuntimeError(f'MixMHC2pred exit {r.returncode}: '
                           f'{(r.stdout + r.stderr)[-2000:]}')
    with open(dst) as fh:
        lines = [ln.rstrip('\n') for ln in fh if not ln.startswith('#')]
    head = lines[0].split('\t')
    col = f'%Rank_{allele}'
    if col not in head:
        raise RuntimeError(f'{col} missing from MixMHC2pred header: {head}')
    ip, ir = head.index('Peptide'), head.index(col)
    out = {}
    for ln in lines[1:]:
        f = ln.split('\t')
        if len(f) <= ir:
            continue
        out[f[ip]] = {'rank': f[ir]}
    if not out:
        raise RuntimeError(f'MixMHC2pred produced no rows for {allele}')
    return out


def main() -> int:
    tool, allele = sys.argv[1], sys.argv[2]
    pep = f'{HERE}/pep/{tool}/{allele}.pep'
    dst = f'{HERE}/out/{tool}/{allele}.csv'
    with open(pep) as fh:
        peps = [ln.strip() for ln in fh if ln.strip()]

    if os.path.exists(dst):
        with open(dst) as fh:
            if sum(1 for _ in fh) - 1 == len(peps):
                print(f'skip {tool}/{allele} ({len(peps)} rows already)')
                return 0

    with tempfile.TemporaryDirectory(prefix=f'{tool}_{allele}_') as tmp:
        got = (run_net if tool == 'net' else run_mix)(pep, allele, tmp)

    keep = NET_KEEP if tool == 'net' else ['rank']
    missing = [p for p in peps if p not in got]
    tmp_dst = dst + '.part'
    with open(tmp_dst, 'w') as fh:
        fh.write('Epi_Seq,' + ','.join(keep) + '\n')
        for p in peps:
            v = got.get(p)
            fh.write(p + ',' + ','.join('' if v is None else v[k] for k in keep) + '\n')
    os.replace(tmp_dst, dst)
    tag = f' MISSING={len(missing)}' if missing else ''
    print(f'{tool}/{allele}: {len(peps)} peptides, {len(got)} scored{tag}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
