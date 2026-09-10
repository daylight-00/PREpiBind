"""HLA_Name -> the allele name each external tool expects.

Our benchmark names a molecule '<beta>_<alpha>', e.g.
'HLA-DRB1*07:01_HLA-DRA*01:01' or 'H2-IAbB_H2-IAbA'. NetMHCIIpan-4.3 and
MixMHC2pred-2.0 each want a different string for the same molecule, and neither
accepts ours:

    molecule                        netMHCIIpan-4.3        MixMHC2pred-2.0
    HLA-DRB1*07:01_HLA-DRA*01:01    DRB1_0701              DRB1_07_01
    HLA-DQB1*03:01_HLA-DQA1*05:05   HLA-DQA10505-DQB10301  DQA1_05_05__DQB1_03_01
    H2-IAbB_H2-IAbA                 H-2-IAb                H2_Aa_b__H2_Ab_b

DR drops the alpha chain on both sides: DRA is effectively invariant and neither
tool takes it. DP/DQ keep both, alpha first.

`check()` regenerates the 125 mappings that 250604/1_ref already had hand-built
and asserts they come out identical, so the derivation is not trusted on its own.
Every generated name is then looked up in the tool's own allele list, which is
where H2-IAg7 is found to be missing from NetMHCIIpan (176 test rows).
"""
from __future__ import annotations

import os
import re

NET_HOME = os.environ.get('NETMHCIIPAN_HOME', '')   # the unpacked netMHCIIpan-4.3 directory
MIX_HOME = os.environ.get('MIXMHC2PRED_HOME', '')   # the unpacked MixMHC2pred-2.0 directory

#: H2 beta-chain tag -> (mix alpha, mix beta) stem. 'IAb' -> A/b, 'IEd' -> E/d.
_H2 = re.compile(r'^H2-I([AE])([a-z0-9]+)$')


def _split(hla_name: str) -> tuple[str, str]:
    """('<beta>', '<alpha>') from our '<beta>_<alpha>' molecule name."""
    parts = hla_name.split('_')
    if len(parts) != 2:
        raise ValueError(f'not a beta_alpha molecule name: {hla_name!r}')
    return parts[0], parts[1]


def _digits(chain: str) -> str:
    """'HLA-DQA1*05:05' -> '0505'"""
    return chain.split('*')[1].replace(':', '')


def _fields(chain: str) -> list[str]:
    """'HLA-DPB1*104:01' -> ['104', '01'].

    Split on the colon, never by character count: DPB1*104:01 has a three-digit
    first field, which is what a d[:2]/d[2:] split gets wrong.
    """
    return chain.split('*')[1].split(':')


def _locus(chain: str) -> str:
    """'HLA-DQA1*05:05' -> 'DQA1'"""
    return chain.split('*')[0].replace('HLA-', '')


def net(hla_name: str) -> str | None:
    beta, alpha = _split(hla_name)
    if beta.startswith('H2-'):
        m = _H2.match(beta[:-1]) or _H2.match(beta.rstrip('B'))
        return f'H-2-I{m.group(1)}{m.group(2)}' if m else None
    if 'DRB' in beta:
        return f'{_locus(beta)}_{_digits(beta)}'
    return f'HLA-{_locus(alpha)}{_digits(alpha)}-{_locus(beta)}{_digits(beta)}'


def mix(hla_name: str) -> str | None:
    beta, alpha = _split(hla_name)
    if beta.startswith('H2-'):
        m = _H2.match(beta.rstrip('B'))
        if not m:
            return None
        gene, hap = m.group(1), m.group(2)
        return f'H2_{gene}a_{hap}__H2_{gene}b_{hap}'
    if 'DRB' in beta:
        return f'{_locus(beta)}_{"_".join(_fields(beta))}'
    return (f'{_locus(alpha)}_{"_".join(_fields(alpha))}__'
            f'{_locus(beta)}_{"_".join(_fields(beta))}')


# ------------------------------------------------------------------ tool support
def net_supported() -> set[str]:
    """Allele names NetMHCIIpan-4.3 has a pseudosequence for."""
    p = os.path.join(NET_HOME, 'data', 'pseudosequence.2023.all.X.dat')
    with open(p) as fh:
        return {ln.split()[0] for ln in fh if ln.strip()}


def mix_supported() -> set[str]:
    """Allele names MixMHC2pred-2.0 has a PWM for."""
    d = os.path.join(MIX_HOME, 'PWMdef')
    return {f[:-4] for f in os.listdir(d)
            if f.endswith('.txt') and not f.startswith('Alleles_list')}


def check(verbose: bool = True) -> None:
    """Reproduce the 125 hand-built mappings from 250604/1_ref and compare."""
    import pandas as pd
    R = os.environ.get('REF_TOOL_ROOT', '')
    n = (pd.read_csv(R + 'test_filtered_netmhcpan_speed.csv')
           .rename(columns={'HLA_Name_x': 'HLA_Name'}))
    m = pd.read_csv(R + 'test_filtered_mixmhcpred_speed.csv')
    for tool, fn, df in (('net', net, n), ('mix', mix, m)):
        ref = df[['HLA_Name', 'HLA_mapped']].drop_duplicates()
        got = ref['HLA_Name'].map(fn)
        bad = ref[got != ref['HLA_mapped']]
        if len(bad):
            raise AssertionError(f'{tool}: {len(bad)} mappings differ\n'
                                 f'{bad.assign(generated=got[bad.index]).to_string(index=False)}')
        if verbose:
            print(f'{tool}: all {len(ref)} hand-built mappings reproduced')


if __name__ == '__main__':
    check()
