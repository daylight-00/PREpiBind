# Supplementary data

Machine-readable versions of the numbers behind the paper. The manuscript prints means; these carry
the per-allele and per-molecule values those means are taken over, the per-seed spread, the
statistics, and the key that maps internal model identifiers to the names used in the text.

Nothing here is recomputed. Every file is a re-packaging of `analysis/scoring/*`, produced by

    python supplementary_data/build_supplementary_data.py            # write
    python supplementary_data/build_supplementary_data.py --check    # regenerate and diff

Numbered `D01..D12`, not `S1..`, because the manuscript owns the Supplementary Table numbering
S1-S19. Those tables are typeset in the supplementary PDF; these are data files.

| file | rows | what it is | where the paper uses it |
|---|---:|---|---|
| `D01_pooled_benchmark` | 59 | Pooled performance of every method on Qualitative, MS, both IC50 thresholds and H2-out: all five metrics with seed-level SD | Table 2, Figure 2 |
| `D02_seed_level_benchmark` | 159 | The three per-seed values behind every mean in D01, each already averaged over its five folds | the `± SD` in Table 2 |
| `D03_per_allele_qualitative` | 395 | Per-allele ROC-AUC on the Qualitative test set, 106 alleles x 7 methods, with per-allele positive and negative counts | Figure 3a shows the distribution; the text quotes only the mean over the 57 shared alleles |
| `D04_per_allele_ms` | 286 | The same for the MS test set | **not in the paper** |
| `D05_per_molecule_lomo` | 396 | Leave-one-molecule-out, per withheld molecule, all five metrics | Figure 3b shows the distribution; the text quotes means |
| `D06_per_molecule_h2_out` | 55 | H2-out per murine molecule. 8 molecules x 7 methods = 55, not 56: NetMHCIIpan-4.3 has no H2-IAg7 pseudosequence | Figure 3d, Supplementary 1.6 |
| `D07_per_serotype` | 80 | Serotype-level performance (HLA-DP, -DQ, -DR, murine H2) on Qualitative and MS | Figure 4, Supplementary 1.6 |
| `D08_stratified_9mer_overlap` | 65 | Qualitative performance stratified by 9-mer overlap with the training epitopes | Supplementary Table S3 |
| `D09_paired_wilcoxon` | 46 | Holm-corrected paired Wilcoxon at three units: the shared allele pair, the beta chain, and the beta chain under LOMO | Supplementary Tables S18, S19, S20 |
| `D10_bootstrap_ci` | 11 | Bootstrap 95% CIs on pooled Qualitative ROC-AUC, marginal and paired | Supplementary Table S11 |
| `D11_reference_tool_allele_map` | 149 | Allele-name mapping and support status for NetMHCIIpan-4.3 and MixMHC2pred-2.0 | Supplementary Text 1.9 |
| `D12_method_key` | 13 | Internal model key, display name, representation family, pinned learning rate, parameter count | — |

## Columns

Shared across the performance tables:

| column | meaning |
|---|---|
| `method` | display name, as used in the paper. `D12` maps it to the internal key |
| `roc_auc`, `pr_auc`, `f1`, `accuracy`, `mcc` | metrics. F1, accuracy and MCC are at a 0.5 probability threshold and are not defined for the two percentile-rank tools |
| `*_sd` | standard deviation over the three seed-level means, `ddof=1`. Not over the 15 fold-seed values, which are not independent |
| `n_seeds`, `n_folds` | how many went into the mean. Always 3 and 5 for the retrained representations |
| `n` | test rows scored. Differs between methods where a tool does not support every allele |
| `HLA_Name`, `beta`, `molecule` | the evaluation unit. `beta` is the beta chain alone, the unit used wherever DeepNeo is compared |
| `HLA_Type` | serotype: HLA-DP, HLA-DQ, HLA-DR or H2 |
| `pos_count`, `neg_count` | class counts for that allele or molecule |
| `evaluation`, `dataset`, `serotype`, `stratum` | which slice the row belongs to |

`D09`: `a` and `b` are the two methods, `median_delta` is the median paired difference `a - b`,
`p` is raw and `p_adj` is Holm-corrected within the panel, `sig` is the conventional star notation.

`D10`: `kind` is `marginal` for a single method's interval and a comparison otherwise; `value` is
the point estimate and `lo`/`hi` the 95% bounds from 1,000 resamples of the test set, recomputing
the folds-then-seeds aggregation for each replicate.

## Aggregation

One rule everywhere, and no seed was selected: average the five folds within a seed to get one
value per seed, then average the three seed values. Reported SD is over those three.
