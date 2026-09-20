"""Leave-one-molecule-out evaluation.

DeepNeo holds out beta chains (38 molecules) while the other models hold out
alpha/beta pairs (47), because DeepNeo does not model the alpha chain. The two
sets map 1:1 onto 38 beta chains, so the comparison is sound, but the n differs.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rawpath as rp
import pipeline as pl


def _test(r):
    base = '250529/lomo_beta' if r.model == 'deepneo' else '250513/lomo'
    return rp.at(f'{base}/{r.group}/test.csv')


#: roots holding the beta-chain LOMO split, which is the only one deepneo is run on
BETA_ROOTS = ('250529/1_lomo_beta', '260829/lomo')


def _rows(df):
    # deepneo is evaluated only on the beta-chain LOMO split
    beta = df['root'].isin(BETA_ROOTS)
    return df[(beta & (df['model'] == 'deepneo')) | (~beta & (df['model'] != 'deepneo'))]


CONFIG = dict(
    name='4_lomo',
    roots=['250513/2_lomo', '250513/6_lomo_2', '250513/4_lomo_chai',
           '250513/7_lomo_chai', '250513/8_lomo_etc', '250524/1_lomo',
           # deepneo e3_s128 for the 30 molecules that seed never reached
           '250529/1_lomo_beta', '260829/lomo',
           # BLOSUM62 recomputed with the fixed encoder; pl.use_rerun_blosum below
           # drops the superseded published cells. See pipeline.BLOSUM_RERUN_ROOT.
           '260905/9_blosum_rerun/2_lomo'],
    # models_* is included deliberately. 64 prediction files were written next to
    # their checkpoints instead of into plots_, and every one of them is the only
    # copy of its (model, molecule, s42, fold): chai for 3 molecules and
    # esmc_small for 10. Excluding them is what left those cells at 2 seeds.
    # collect() drops any models_ row a plots_ row already covers, so nothing is
    # double-counted - here that overlap is empty.
    dir_filter=('plots_*', 'models_*'),
    test_for=_test,
    row_filter=lambda df: pl.use_rerun_blosum(_rows(df)),
    # No seed is selected. All three seeds are kept and reduced folds-then-seeds;
    # see pipeline.aggregate(). The learning rate is pinned instead of chosen,
    # because choosing either one on the test metric is selection on the number
    # being reported.
    group_name='molecule',
    group_keys=['group'],
    lr_filter=pl.pin_lr,         # exactly one lr per model; pinned for the guard
    ref=True,
)
