"""Qualitative test set stratified by 9-mer overlap with the training epitopes.

This is the analysis behind `tab:supp-stratified` and the main-text sentence
"BLOSUM62 showed roughly twice the ROC-AUC drop" (oup:358). It used to be an ad-hoc
recomputation with no code in this tree, which is why it could not be re-derived when the BLOSUM62
baseline was re-run. It lives here now.

## The strata

An epitope is scored against the union of the training epitopes:

    exact 15-mer match        the epitope string occurs verbatim in train
    9-mer overlap, not exact  not exact, but shares at least one 9-mer with train
    >=1 9-mer overlap         the union of those two
    no 9-mer overlap          neither

## The definition is confirmed, not assumed

Recovered by matching the published row counts exactly, in both test sets:

    stratum                    test.csv   test_beta.csv   published range
    exact 15-mer match           36,146          35,473   35,473-36,146
    9-mer overlap, not exact      6,978           6,971    6,971- 6,978
    >=1 9-mer overlap            43,124          42,444   42,444-43,124
    no 9-mer overlap              5,228           5,222    5,222- 5,228
    overall                      48,352          47,666   47,666-48,352

All eight agree, so the stratum boundaries are the published ones and not a
reconstruction that merely looks plausible. The two columns are the two test sets:
DeepNeo is scored on test_beta.csv, every other model on test.csv, which is where
the published ranges come from.

`full/train.csv` and `full/train_beta.csv` hold the same 46,487 unique epitopes and
therefore the same 203,206 9-mers, so one reference set serves both.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import functools
import numpy as np
import pandas as pd
import rawpath as rp
import pipeline as pl

K = 9


@functools.lru_cache(maxsize=1)
def _train_reference():
    """(exact epitope set, 9-mer set) of the training epitopes."""
    tr = pd.read_csv(rp.data('dataset/full/train.csv'))['Epi_Seq'].astype(str)
    exact = set(tr)
    nine = set()
    for s in exact:
        nine.update(s[i:i + K] for i in range(len(s) - K + 1))
    return exact, nine


#: epitope -> (exact, has_nine). Memoised across calls: pipeline.add_metrics scores
#: every prediction file against each of the five strata, so without this the same
#: 48,352 epitopes are re-tested five times per file and the analysis takes ~25 min
#: instead of ~1. The test sets share almost all of their epitopes, so one dict
#: serves both.
_FLAG_CACHE: dict[str, tuple[bool, bool]] = {}


def _flag_one(s: str) -> tuple[bool, bool]:
    v = _FLAG_CACHE.get(s)
    if v is None:
        exact_set, nine_set = _train_reference()
        v = _FLAG_CACHE[s] = (
            s in exact_set,
            bool({s[i:i + K] for i in range(len(s) - K + 1)} & nine_set),
        )
    return v


def _flags(frame: pd.DataFrame) -> pd.DataFrame:
    pairs = [_flag_one(s) for s in frame['Epi_Seq'].astype(str)]
    return pd.DataFrame(pairs, columns=['exact', 'nine'], index=frame.index)


# Masks return a numpy bool ARRAY, never a Series. pipeline.ref_level() applies them
# to the reference-tool join, which is EMPTY for test_beta.csv (the tools' rows are
# keyed by pair name, so nothing matches a beta-only test set). An empty pandas
# boolean Series comes back as dtype object, and `frame[object_series]` is column
# selection, not row selection -- it silently returned a frame with zero columns and
# the next line failed with KeyError: 'Target'. A bool array is unambiguous.
def _m_all(frame):
    return np.ones(len(frame), dtype=bool)


def _m_no_overlap(frame):
    f = _flags(frame); return (~f['nine'] & ~f['exact']).to_numpy(dtype=bool)


def _m_any_overlap(frame):
    f = _flags(frame); return (f['nine'] | f['exact']).to_numpy(dtype=bool)


def _m_exact(frame):
    return _flags(frame)['exact'].to_numpy(dtype=bool)


def _m_overlap_not_exact(frame):
    f = _flags(frame); return (f['nine'] & ~f['exact']).to_numpy(dtype=bool)


def _test(r):
    return rp.data(f'dataset/full/test{"_beta" if r.model == "deepneo" else ""}.csv')


def _rows(df):
    # same as 1_whole: the 1_bulk runs of deepneo were superseded
    return df[~(df['root'].str.contains('1_bulk') & (df['model'] == 'deepneo'))]


#: 'Overall test set' is the unstratified row; pipeline emits it as sero='Overall'
SUBSETS = [
    ('Overall', _m_all),
    ('No 9-mer overlap', _m_no_overlap),
    ('>=1 9-mer overlap', _m_any_overlap),
    ('Exact 15-mer match', _m_exact),
    ('9-mer overlap, not exact', _m_overlap_not_exact),
]

CONFIG = dict(
    name='8_strat',
    # must match 1_whole: the stratification is that analysis's test set, sliced
    roots=['250513/1_bulk', '250516/7_ic50_etc', '250524/0_bulk',
           # BLOSUM62 recomputed with the fixed encoder; pl.use_rerun_blosum below
           # drops the superseded published cells. See pipeline.BLOSUM_RERUN_ROOT.
           '260905/9_blosum_rerun/1_bulk'],
    dir_filter='plots',
    test_for=_test,
    subsets=SUBSETS,
    row_filter=lambda df: pl.use_rerun_blosum(_rows(df)),
    slice_name='stratum',
    group_keys=['sero'],
    lr_filter=pl.pin_lr,
    ref=True,
)
