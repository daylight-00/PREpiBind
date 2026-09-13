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
needed for the second. `make figures` regenerates fig2-fig5 and figS1-figS2; fig1 is a drawing and
is shipped as a file.

## The snapshot mirrors the scratch path

```
0_raw/250513/2_lomo/plots_HLA-DRB1~15:01_HLA-DRA~01:01/e5_s100/pred-esmc_small_fold0.csv
     └────────────────────── identical to the path under /home/hwjang/project ──┘
```

So a file's provenance *is* its path: `rp.source(rel)` is `SCRATCH / rel`, and
`rp.find(abs_path)` goes back the other way.

This matters. The snapshot previously flattened several scratch directories into one
`<lr>_<seed>` level. Because `250513/1_bulk/1/plots/` and `250513/1_bulk/4/plots/`
use the same seed directory names, they overwrote each other: 194 of the files the
analyses need were missing or replaced, and nothing reported it. Mirroring makes
that impossible by construction.

### Clusters

| alias | role | relevance here |
| --- | --- | --- |
| `abc` | ABO Cluster | produced every training/inference run in the snapshot |
| `victor` | shared workstation | serves the official prepibind-web instance |
| `alpha` | personal workstation | ran the dataset-organisation step |

The scratch root is `/home/hwjang/project` on all three, so `src_cluster` is a
single value (`abc`) and `victor`/`alpha` only consume the snapshot. The one
exception is the three `hum_ani_full.csv` marked `lost`: dataset organisation ran
on `alpha`, and they exist nowhere on `abc`. `tools/sync_snapshot.py` already
reserves their paths, so copying them from `alpha` and re-syncing picks them up
with no code change.

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

`raw_manifest.csv` also records what is deliberately *not* in the snapshot:

| state | meaning |
| --- | --- |
| `snapshot` | the file is in `0_raw/` |
| `external` | too large to ship (the four HLA embedding `.h5`, 1.3 GB total). `figures/figS2_umap.py` reproduces its figure from `figures/umap_cache/*.npz` instead. |
| `lost` | the original is gone. Three `hum_ani_full.csv` files referenced by `figures/` were deleted from scratch before this index existed; they were recovered and now ship in `figures/data/figure_inputs.tar.zst`. |

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
pinned to (`lr_filter`), and any externally published baseline rows (`extra_rows`).

The same logic used to be copy-pasted into all seven notebooks, at 0.55-0.98
pairwise similarity. That is how `header=None` came to be missing from exactly one
copy - shifting every prediction by one row and depressing whole-set AUC by ~0.15 -
and how `4_lomo` came to lack the `dir` filter the other analyses had.

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

This replaces the old `select_best()`, which took the best seed per fold - first by
test ROC-AUC, later by `val_loss`. Selecting on the test metric inflates the
reported number by a model-dependent amount; selecting on validation fixed the leak
but still reported a maximum rather than a typical result. Reporting the mean and
its spread across seeds answers the question the benchmark is actually asking.

Allele-wise and LOMO results reduce the same way and stop at **one value per allele
or per withheld molecule**. That value, never the individual run, is the unit for
box plots and for statistics - 15 runs of one model on one allele are not 15
independent observations.

Checkpoint selection *within* a run is unchanged, and is not free to change: it is
the epoch with the lowest validation loss, which is what `prepibind/train.py` saved. The
`Val ROC-AUC` in `train.log` cannot be used, because train.py calls its own
`roc_auc_score` with the arguments swapped and 21% of the logged values fall
outside `[0,1]`. See `tools/parse_train_logs.py`.

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

The old notebooks instead used Mann-Whitney U whenever DeepNeo was involved and

```python
wilcoxon(a.iloc[:min_len], b.iloc[:min_len])
```

for everything else - a paired test that pairs by *row position* and drops the tail.
`figures/tab2_pooled_benchmark.ipynb` reports the collapse alongside two alternatives (the full 47
pairs without DeepNeo, and the 32 betas with exactly one pairing); all three agree on
every conclusion.

## Two guards that exist because their absence caused silent wrong numbers

- `read_pred()` always passes `header=None`. These files have no header row, so
  without it pandas consumes the first prediction as a column name.
- `add_metrics()` treats a prediction/test length mismatch as an error. The old code
  merged on index, which is an inner join, so a mismatched pair silently scored on
  whichever rows lined up. `3_ic` was pairing 47666 full-dataset predictions with a
  14117-row ic50 test set and reporting roc_auc around 0.51.
- `pin_one_lr()` raises if more than one learning rate survives `lr_filter`. The old
  code broke that tie with the test metric, which is the same leak as seed selection
  one level up.

## Moving the snapshot between clusters

```bash
tools/snapshot.sh pack                # -> 0_raw.tar.zst  (~273 MB)
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
