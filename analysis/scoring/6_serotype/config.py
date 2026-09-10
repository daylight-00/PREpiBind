"""Per-serotype (DR / DP / DQ / H2) breakdown of the full comparison."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rawpath as rp
import pipeline as pl


def _test(r):
    return rp.data(f'dataset/full/test{"_beta" if r.model == "deepneo" else ""}.csv')


def _rows(df):
    return df[~(df['root'].str.contains('1_bulk') & (df['model'] == 'deepneo'))]


CONFIG = dict(
    name='6_serotype',
    # must match 1_whole: without 250524/0_bulk the breakdown silently lacks
    # ESM3 Large, the row the PI asked about
    roots=['250513/1_bulk', '250516/7_ic50_etc', '250524/0_bulk',
           # BLOSUM62 recomputed with the fixed encoder; pl.use_rerun_blosum below
           # drops the superseded published cells. See pipeline.BLOSUM_RERUN_ROOT.
           '260905/9_blosum_rerun/1_bulk'],
    dir_filter='plots',
    test_for=_test,
    subsets=['DR', 'DP', 'DQ', 'H2'],
    row_filter=lambda df: pl.use_rerun_blosum(_rows(df)),
    # Nothing is inherited any more: with no seed selection there is no choice to
    # copy from 1_whole. Every (seed, fold) run is scored and aggregated the same
    # way here, so the serotype breakdown and its parent see the same runs.
    slice_name='serotype',
    group_keys=['sero'],
    lr_filter=pl.pin_lr,
    ref=True,
)
