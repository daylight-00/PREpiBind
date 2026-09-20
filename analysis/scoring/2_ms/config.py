"""Mass-spectrometry (ms_ql) comparison."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rawpath as rp
import pipeline as pl


def _test(r):
    return rp.data(f'dataset/ms_ql/test{"_beta" if r.model == "deepneo" else ""}.csv')


CONFIG = dict(
    name='2_ms',
    roots=['250527/1_ms_re', '250527/2_ms_esm3_re',
           # e5_s128 for chai and esmc_small, and the esmc_small e5_s100 fold
           # whose checkpoint had survived but whose inference never ran
           '260829/ms',
           # af3, boltz, esmc_large, esmc_medium, esm3_medium and esm3_large -
           # the tab:perf-summary MS blanks. esm3_large's 15 cells landed
           # 2026-09-03, so the row_filter that held it out is gone.
           '260830/1_ms',
           # BLOSUM62 recomputed with the fixed encoder; pl.use_rerun_blosum below
           # drops the superseded published cells. See pipeline.BLOSUM_RERUN_ROOT.
           '260905/9_blosum_rerun/5_ms'],
    dir_filter='plots*',
    test_for=_test,
    row_filter=pl.use_rerun_blosum,
    # No seed is selected. All three seeds are kept and reduced folds-then-seeds;
    # see pipeline.aggregate(). The learning rate is pinned instead of chosen,
    # because choosing either one on the test metric is selection on the number
    # being reported.
    group_keys=['dir'],
    lr_filter=pl.pin_lr,
    allele_table=True,         # one lr per model here already; pinned for the guard
    # NetMHCIIpan-4.3 / MixMHC2pred-2.0, scored by pipeline.ref_level() on the
    # ms_ql rows above rather than typed in. The scalars that used to sit here
    # (0.9718296579610518 / 0.950912204811488) were measured on ms_ic, which is a
    # different 20,627-row split, and the file behind them had been lost - the
    # mismatch that prompted the reference-tool rescoring in scoring/0_ref/.
    ref=True,
)
