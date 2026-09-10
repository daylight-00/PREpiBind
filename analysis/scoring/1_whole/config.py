"""Whole-test-set comparison across all models (full dataset)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rawpath as rp
import pipeline as pl


def _test(r):
    return rp.data(f'dataset/full/test{"_beta" if r.model == "deepneo" else ""}.csv')


def _rows(df):
    # deepneo on the full set comes from 7_ic50_etc/plots; the 1_bulk runs of it
    # were superseded, so they are dropped here as the notebook always did.
    return df[~(df['root'].str.contains('1_bulk') & (df['model'] == 'deepneo'))]


CONFIG = dict(
    name='1_whole',
    roots=['250513/1_bulk', '250516/7_ic50_etc', '250524/0_bulk',
           # BLOSUM62 recomputed with the fixed encoder; pl.use_rerun_blosum below
           # drops the superseded published cells. See pipeline.BLOSUM_RERUN_ROOT.
           '260905/9_blosum_rerun/1_bulk'],
    dir_filter='plots',          # bare 'plots' only: plots_500/_1000/_score are 3_ic
    test_for=_test,
    row_filter=lambda df: pl.use_rerun_blosum(_rows(df)),
    # No seed is selected. All three seeds are kept and reduced folds-then-seeds;
    # see pipeline.aggregate(). The learning rate is pinned instead of chosen,
    # because choosing either one on the test metric is selection on the number
    # being reported.
    group_keys=[],
    lr_filter=pl.pin_lr,
    allele_table=True,
    ref=True,
)
