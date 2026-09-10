"""Human/animal (H2) out-of-distribution evaluation."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rawpath as rp
import pipeline as pl

CONFIG = dict(
    name='5_h2',
    roots=['250513/5_hum_ani', '250513/9_hum_ani_etc', '250524/2_h2',
           # BLOSUM62 recomputed with the fixed encoder; pl.use_rerun_blosum below
           # drops the superseded published cells. See pipeline.BLOSUM_RERUN_ROOT.
           '260905/9_blosum_rerun/3_humani'],
    dir_filter='plot_ani',   # 9_hum_ani_etc/plot/ holds full-dataset predictions
    test_for=lambda r: rp.at('250513/5_hum_ani/ani_full.csv'),
    row_filter=pl.use_rerun_blosum,
    # No seed is selected. All three seeds are kept and reduced folds-then-seeds;
    # see pipeline.aggregate(). The learning rate is pinned instead of chosen,
    # because choosing either one on the test metric is selection on the number
    # being reported.
    group_keys=[],
    lr_filter=pl.pin_lr,
    allele_table=True,
    ref=True,
)
