"""Per-serotype (DR / DP / DQ / H2) breakdown of the ms_ql comparison."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rawpath as rp
import pipeline as pl


def _test(r):
    return rp.data(f'dataset/ms_ql/test{"_beta" if r.model == "deepneo" else ""}.csv')


def _rows(df):
    return df[~(df['root'].str.contains('1_bulk') & (df['model'] == 'deepneo'))]


CONFIG = dict(
    name='7_serotype_ms',
    # must match 2_ms exactly, so both see the same runs
    roots=['250527/1_ms_re', '250527/2_ms_esm3_re', '260829/ms',
           # BLOSUM62 recomputed with the fixed encoder; pl.use_rerun_blosum below
           # drops the superseded published cells. See pipeline.BLOSUM_RERUN_ROOT.
           '260905/9_blosum_rerun/5_ms'],
    dir_filter='plots*',
    test_for=_test,
    row_filter=pl.use_rerun_blosum,
    subsets=['DR', 'DP', 'DQ', 'H2'],
    
    # Nothing is inherited any more: with no seed selection there is no choice to
    # copy from 2_ms. Every (seed, fold) run is scored and aggregated the same
    # way here, so the serotype breakdown and its parent see the same runs.
    slice_name='serotype',
    group_keys=['sero'],
    lr_filter=pl.pin_lr,
    # NetMHCIIpan-4.3 / MixMHC2pred-2.0 per serotype, scored by
    # pipeline.ref_level() on the ms_ql rows above. The eight scalars that used to
    # sit here came from 250714/10_ms/6_serotype/metrics_results_max_.csv, were
    # measured on ms_ic rather than ms_ql, and were hand-added so they were lost on
    # any regeneration.
    ref=True,
)
