"""IC50 threshold comparison, 500 and 1000 nM."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rawpath as rp
import pipeline as pl


def _test(r):
    return rp.data(f'dataset/ic50/test{"_beta" if r.model == "deepneo" else ""}.csv')


def _target(r):
    return 'Target_1000' if '1000' in r.dir else 'Target_500'


CONFIG = dict(
    name='3_ic',
    # 'plots_*' deliberately excludes the bare 'plots' directory: those predictions
    # were made against the full dataset and belong to 1_whole. The old notebook
    # collected them, paired them with the ic50 test set, and the index merge
    # truncated 47666 predictions to 14117 rows, yielding roc_auc around 0.51.
    roots=['250516/7_ic50_etc', '250524/4_ic50',
           # af3, boltz, esmc_large, esmc_medium, esm3_medium at both
           # thresholds - the tab:perf-summary IC50 blanks - and esm3_large,
           # admitted 2026-09-03. plots_500 is complete at 15/15; plots_1000 is
           # NOT (s42 5 folds, s100 4, s128 1 at the time of writing), so
           # coverage.csv carries three rows for it and the aggregate there
           # averages incomplete seeds. The manuscript must take esm3_large's
           # IC50<1000 value from seed_level_results.csv, seed s42 only, for as
           # long as that arm's remaining folds are incomplete.
           '260830/2_ic50',
           # BLOSUM62 recomputed with the fixed encoder; pl.use_rerun_blosum below
           # drops the superseded published cells. See pipeline.BLOSUM_RERUN_ROOT.
           '260905/9_blosum_rerun/4_ic50'],
    # plots_500 and plots_1000 only. A third variant, plots_score, regressed the
    # 1-log50k(nM) transform of log_IC50 instead of a binarised label; it covers 6 of the
    # 13 methods, appears nowhere in the paper, and its target column has been dropped from
    # the released dataset, so it is excluded rather than shipped half-finished.
    dir_filter=('plots_500', 'plots_1000'),
    test_for=_test,
    row_filter=pl.use_rerun_blosum,
    target_for=_target,
    # No seed is selected. All three seeds are kept and reduced folds-then-seeds;
    # see pipeline.aggregate(). The learning rate is pinned instead of chosen,
    # because choosing either one on the test metric is selection on the number
    # being reported.
    group_keys=['dir'],
    # esmc_small has both e4 and e5 under plots_score. The old code kept whichever
    # scored higher on the test set; pin_lr takes e5, the rate every other
    # non-deepneo model uses.
    lr_filter=pl.pin_lr,
    ref=True,
    # NetMHCIIpan-4.3 is scored on its BA head here, not the EL head every other
    # analysis uses. The two heads have different training targets - EL is the
    # eluted-ligand/immunopeptidomics model, BA is fitted to IEDB measured IC50 -
    # and this test set is measured IC50 binarised at 500 / 1000 nM, so BA is the
    # matching output. Scoring it with EL cost the tool 0.18-0.20 ROC-AUC
    # (0.730 -> 0.909 at <500, 0.717 -> 0.899 at <1000), in the direction that
    # flatters the representations beside it. ba_rank rather than ba_score, so that
    # the -%Rank convention and the within-allele calibration caveat hold for every
    # reference cell alike. MixMHC2pred-2.0 has no affinity head, so its row is
    # unchanged and stays a ligand-likelihood %Rank.
    ref_col={'netmhcpan': 'ba_rank'},
)
