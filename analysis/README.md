# analysis

Everything behind the paper's figures: the raw model outputs, the per-analysis
scoring, and the plotting notebooks.

The goal is that **all of it runs on a cluster that has no experiment data**, while
staying able to answer "where did this number come from?" for any file.

```
analysis/
├── 0_raw/              prediction snapshot (not in git, ~1.3 GB)
├── raw_manifest.csv    the index: integrity + provenance for every file (tracked)
├── rawpath.py          path resolver used by every analysis
├── tools/
│   ├── sync_snapshot.py    populate 0_raw/ from scratch
│   ├── build_manifest.py   regenerate raw_manifest.csv
│   └── snapshot.sh         pack / unpack / verify for cluster transfer
├── scoring/
│   ├── pipeline.py         shared collect -> score -> aggregate -> test logic
│   ├── run.py              run one analysis that has no notebook (8_strat)
│   └── <analysis>/
│       ├── config.py       what makes this analysis different
│       └── anal_pred_*.ipynb
└── figures/
    ├── figio.py            shared loading / units / annotation for the figures
    ├── fig1.{pdf,svg}      the workflow and architecture schematic. Drawn by hand, NOT produced
    │                       by `make figures`; the editable source is
    │                       fig1_workflow_and_architecture.fig
    ├── umap_cache/         the UMAP coordinates behind figS2, so it redraws without a GPU
    └── *.ipynb             figure notebooks, reading scoring/*/*_results.csv
```

`make figures` and `make scoring` at the repository root drive both halves; the snapshot is only
needed for the second. `make figures` regenerates fig2-fig5, figS1-figS2, Table 2 and the paired
tests, plus `figures/supp_ref_tables.tex` and the `supp_ref_*.csv` behind D09; fig1 is a drawing and
is shipped as a file.

The rebuild is comparable to the committed PDFs, not merely equivalent. From the locked environment
all six come out pixel-identical at 150 dpi; the five matplotlib ones are byte-identical once
`/CreationDate`, `/ModDate`, `/ID` and `/Producer` are stripped, and figS1, which cairosvg writes,
is byte-identical in every object except the one holding its own `/CreationDate` — cairo packs that
into a compressed object stream, so stripping the plain-text keys does not reach it. The page
content is unchanged.

This is why `pyproject.toml` pins matplotlib: a different minor version redraws the same figures
with different sub-pixel layout and tight-bbox rounding, and gives figS2 538 vector objects instead
of 555. The `.svg` companions are a different matter — they carry a generation timestamp that
nothing strips, so `make figures` always leaves them modified in `git status`.

`make scoring` aborts unless `PREPIBIND_RAW_ROOT` points at the unpacked snapshot; `rawpath` falls
back to `IMG_RAW_ROOT`, then to `0_raw/` beside it. The snapshot is in no clone and is not yet
downloadable: the archival deposit that will carry it is staged but has no host and no DOI, see
`tools/release/deposit/README.md`.

## The snapshot mirrors the scratch path

```
0_raw/250513/2_lomo/plots_HLA-DRB1~15:01_HLA-DRA~01:01/e5_s100/pred-esmc_small_fold0.csv
     └──────────────────────── identical to the path under the scratch root ────┘
```

So a file's provenance *is* its path: `rp.source(rel)` is `SCRATCH / rel`, and
`rp.find(abs_path)` goes back the other way.

Mirroring rather than flattening is deliberate: directories that share seed names at different
scratch roots overwrite each other under a flattened layout, silently dropping files no one
notices.

## Reading data

Analyses never build absolute paths. They go through `rawpath`:

```python
import rawpath as rp

rp.at('250513/2_lomo/plots_X/e5_s100/pred-esmc_small_fold0.csv')  # snapshot
rp.at('250513/lomo/HLA-DRB1~15:01_HLA-DRA~01:01/test.csv')        # inputs too
rp.data('dataset/full/test.csv')   # the repo's own data/, not the snapshot
rp.source(rel)                     # where it came from
rp.find(abs_path)                  # reverse lookup
rp.verify(rel)                     # md5 against the manifest
rp.table(root='250513/2_lomo')     # manifest slice as a DataFrame
rp.status()                        # what is visible on this cluster
```

`raw_manifest.csv` gives every file a state:

| state | meaning |
| --- | --- |
| `snapshot` | the file is in `0_raw/` |
| `external` | too large to ship (the four HLA embedding `.h5`, 1.3 GB total). `figures/figS2_umap.py` reproduces its figure from `figures/umap_cache/*.npz` instead. |
| `bundled` | small enough to ship with the repository. The three `hum_ani_full.csv` that `figures/figS1_dataset_overlap.ipynb` reads are in `figures/data/figure_inputs.tar.zst`, so `rp.at()` resolves them and the figure layer needs no snapshot. |

## Running an analysis

Each analysis is one config plus the shared pipeline:

```python
import sys; sys.path.insert(0, '..')
import pipeline as pl
from config import CONFIG

per_run, seed_level, rep_level = pl.run(CONFIG)
```

`config.py` declares only what differs: which scratch roots to read, which
directories within them (`dir_filter`), which test set each row is scored against
(`test_for`), which label column (`target_for`), which learning rate the models are
pinned to (`lr_filter`), and whether the published tools are scored beside the
representations (`ref=True`: `pipeline.ref_level()` scores NetMHCIIpan-4.3 and
MixMHC2pred-2.0 from `0_raw/260830/ref/` on this analysis's own test set).

One implementation, not seven copies: the notebooks differ only in their `config.py`.

## How results are aggregated

**No seed is selected.** Every `(seed, fold)` run a representation has is kept, and
reduced in two steps:

```
5 folds
   |  mean
1 value per seed
   |  mean +- SD across 3 seeds
representation-level performance
```

The SD is therefore the spread across three independent training runs, taken from
three numbers with `ddof=1`. It is deliberately *not* the SD of all 15 fold/seed
values: folds share a test set and are not independent replicates, so pooling them
understates the variance that matters.

Selecting a best seed is what this replaces: on the test metric it leaks, and on validation it
still reports a maximum rather than a typical result.

Allele-wise and LOMO results reduce the same way and stop at **one value per allele
or per withheld molecule**. That value, never the individual run, is the unit for
box plots and for statistics - 15 runs of one model on one allele are not 15
independent observations.

Checkpoint selection *within* a run is the epoch with the lowest validation loss, which is what
`prepibind/train.py` saved. The `Val ROC-AUC` in `train.log` is unusable: `train.py` calls
`roc_auc_score` with its arguments swapped, and 21 % of the logged values fall outside [0, 1]. See
`tools/parse_train_logs.py`.

### Outputs

`pl.run()` writes, per analysis:

| file | one row per |
| --- | --- |
| `per_run_results.csv` | `(model, seed, fold, group)` - every run scored |
| `seed_level_results.csv` | `(model, seed, group)` - folds averaged |
| `representation_level_results.csv` | `(model, group)` - seeds averaged, with `*_sd`, `n_seeds` |
| `coverage.csv` | a hole in the seeds x folds grid. **Empty is the expected state.** |
| `allele_level_results.csv` | `(model, allele)`, where `allele_table=True` |
| `lomo_level_results.csv` | `(model, withheld molecule)`, for `4_lomo` |

Aggregation is reproducible from `per_run_results.csv` alone.

### Comparing against DeepNeo

DeepNeo does not model the alpha chain, so it holds out **beta chains** where the
other models hold out **alpha/beta pairs**: 48 units against 58 allele-wise, 38
against 47 in LOMO, and the two sides share no unit names at all. Several beta
chains carry two to four alpha pairings.

`pipeline.collapse_to_beta()` averages the pair-split values within each beta, which
is what makes the five methods commensurable - the beta is the only unit all of them
have. Everything downstream is then a fully paired Wilcoxon signed-rank test,
Holm-corrected across the pairs in a panel.

`figures/paired_tests.ipynb` reports the collapse alongside two alternatives — the full
47 pairs without DeepNeo, and the 32 betas with exactly one pairing — and all three agree on every
conclusion.

## Three guards, each against a silent wrong number

- `read_pred()` always passes `header=None`. These files have no header row, so without it pandas
  consumes the first prediction as a column name.
- `add_metrics()` treats a prediction/test length mismatch as an error rather than merging on index,
  which is an inner join and would score on whichever rows happen to line up.
- `pin_one_lr()` raises if more than one learning rate survives `lr_filter`, rather than breaking the
  tie with the test metric.

## Moving the snapshot between machines

```bash
tools/snapshot.sh pack                # -> 0_raw.tar.zst  (~308 MB)
# copy it across, then on the other cluster:
tools/snapshot.sh unpack 0_raw.tar.zst
tools/snapshot.sh verify              # md5 every file against the manifest
```

`sync_snapshot.py` already leaves out the per-fold ROC/PR diagnostic PNGs
(8850 files, 271 MB): no analysis reads them and they regenerate from `pred-*` plus
the test sets, so omitting them still reproduces every figure. `pack --lean` exists
to drop them from a snapshot that does contain them.

## Adding a new experiment

1. Add its scratch root to `ROOTS` in `tools/sync_snapshot.py` (and any new ground
   truth to `INPUT_GLOBS` - without it the snapshot holds predictions but no labels,
   and no metric can be computed off-cluster).
2. `python tools/sync_snapshot.py`
3. `python tools/build_manifest.py`
4. Point an analysis `config.py` at the new root.

`python tools/build_manifest.py --check` reports drift without writing, and is the
quick way to confirm a snapshot matches its index.
