#!/usr/bin/env python3
"""Build the machine-readable supplementary data set.

The paper prints means. These files carry the per-allele and per-molecule values behind them, plus
the seed-level spread, the statistics and the method key. Nothing here is recomputed: every file is
a re-packaging of analysis/scoring/*, with internal model keys replaced by display names and
absolute paths already stripped upstream.

They are numbered D01..D12 rather than S1.. because the manuscript owns the Supplementary Table
numbering S1-S19; these are a separate, data-only series.

    python supplementary_data/build_supplementary_data.py            # write
    python supplementary_data/build_supplementary_data.py --check    # regenerate and diff
"""
import argparse
import io
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SCORING = os.path.join(REPO, "analysis", "scoring")
FIGURES = os.path.join(REPO, "analysis", "figures")

TEMPLATE = pd.read_csv(os.path.join(FIGURES, "template.csv"))
NAME = dict(zip(TEMPLATE["model"], TEMPLATE["full_name"]))
FAMILY = dict(zip(TEMPLATE["model"], TEMPLATE["fam"]))


def named(df, col="model", keep_group=False):
    """Internal key -> display name, and drop columns that mean nothing outside this repository.

    The scoring tables already name their columns for what they hold; this only drops the ones
    that mean nothing outside the repository.
    """
    out = df.copy()
    if "method" not in out.columns:
        out.insert(0, "method", out[col].map(NAME).fillna(out[col]))
    else:
        # the scoring tables already carry a display name; complete it for the two published tools
        out["method"] = out[col].map(NAME).fillna(out["method"])
        out.insert(0, "method", out.pop("method"))
    drop = [col, "plot", "seeds"] + ([] if keep_group else ["group"])
    return out.drop(columns=[c for c in drop if c in out.columns])


def rep(analysis, **filt):
    d = pd.read_csv(os.path.join(SCORING, analysis, "representation_level_results.csv"))
    for k, v in filt.items():
        d = d[d[k] == v]
    return d


def build():
    out = {}

    # D01  pooled benchmark, every method on every evaluation
    frames = []
    for label, analysis, filt in [
        ("Qualitative", "1_whole", {}),
        ("MS", "2_ms", {}),
        ("IC50 < 500 nM", "3_ic", {"dir": "plots_500"}),
        ("IC50 < 1000 nM", "3_ic", {"dir": "plots_1000"}),
        ("H2-out", "5_h2", {}),
    ]:
        d = named(rep(analysis, **filt))
        d.insert(1, "evaluation", label)
        frames.append(d.drop(columns=[c for c in ("dir", "lr") if c in d.columns]))
    out["D01_pooled_benchmark"] = pd.concat(frames, ignore_index=True)

    # D02  the per-seed values behind every mean in D01
    frames = []
    for label, analysis in [("Qualitative", "1_whole"), ("MS", "2_ms"), ("IC50", "3_ic"), ("H2-out", "5_h2")]:
        p = os.path.join(SCORING, analysis, "seed_level_results.csv")
        if not os.path.exists(p):
            continue
        d = named(pd.read_csv(p))
        d.insert(1, "evaluation", label)
        frames.append(d.drop(columns=[c for c in ("lr",) if c in d.columns]))
    out["D02_seed_level_benchmark"] = pd.concat(frames, ignore_index=True)

    # D03/D04/D06  per-allele and per-molecule ROC-AUC
    for key, analysis in [("D03_per_allele_qualitative", "1_whole"),
                          ("D04_per_allele_ms", "2_ms"),
                          ("D06_per_molecule_h2_out", "5_h2")]:
        d = named(pd.read_csv(os.path.join(SCORING, analysis, "allele_level_results.csv")))
        out[key] = d.rename(columns={"roc_auc": "roc_auc", "roc_auc_sd": "roc_auc_sd"})

    # D05  leave-one-molecule-out, per withheld molecule
    d = named(pd.read_csv(os.path.join(SCORING, "4_lomo", "lomo_level_results.csv")), keep_group=True)
    out["D05_per_molecule_lomo"] = d.drop(
        columns=[c for c in ("lr",) if c in d.columns])

    # D07  serotype level, both datasets
    frames = []
    for label, analysis in [("Qualitative", "6_serotype"), ("MS", "7_serotype_ms")]:
        d = named(rep(analysis))
        d.insert(1, "dataset", label)
        frames.append(d.drop(columns=[c for c in ("lr",) if c in d.columns]))
    out["D07_per_serotype"] = pd.concat(frames, ignore_index=True)

    # D08  9-mer overlap stratification
    d = named(rep("8_strat"))
    out["D08_stratified_9mer_overlap"] = d.drop(columns=[c for c in ("lr",) if c in d.columns])

    # D09  Holm-corrected paired Wilcoxon, both units
    frames = []
    for unit, f in [("allele pair", "supp_ref_pairs.csv"), ("beta chain", "supp_ref_pairs_beta.csv")]:
        d = pd.read_csv(os.path.join(FIGURES, f))
        d.insert(0, "unit", unit)
        frames.append(d)
    out["D09_paired_wilcoxon"] = pd.concat(frames, ignore_index=True)

    # D10  bootstrap confidence intervals on pooled Qualitative ROC-AUC
    d = pd.read_csv(os.path.join(SCORING, "9_boot", "bootstrap_results.csv"))
    for c in ("a", "b"):
        d[c] = d[c].map(NAME).fillna(d[c])
    out["D10_bootstrap_ci"] = d

    # D11  reference-tool allele coverage
    d = pd.read_csv(os.path.join(SCORING, "0_ref", "allele_map.csv"))
    out["D11_reference_tool_allele_map"] = d.rename(columns={
        "net": "netmhciipan_allele", "mix": "mixmhc2pred_allele",
        "net_ok": "netmhciipan_supported", "mix_ok": "mixmhc2pred_supported",
        "n_pep": "n_peptides"})

    # D12  method key
    d = TEMPLATE.drop(columns=["plot"]).rename(columns={
        "model": "model_key", "full_name": "method", "fam": "representation_family",
        "params": "parameters", "lr": "learning_rate"})
    out["D12_method_key"] = d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="regenerate and report differences")
    a = ap.parse_args()
    tables = build()
    bad = 0
    for name, df in sorted(tables.items()):
        p = os.path.join(HERE, f"prepibind_{name}.csv")
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        if a.check:
            if not os.path.exists(p):
                print(f"  MISSING  {os.path.basename(p)}"); bad += 1
            elif open(p).read() != buf.getvalue():
                print(f"  DIFFERS  {os.path.basename(p)}"); bad += 1
            else:
                print(f"  ok       {os.path.basename(p)}  {len(df):>5,} rows")
        else:
            open(p, "w").write(buf.getvalue())
            print(f"  {os.path.basename(p):<44s} {len(df):>5,} rows x {len(df.columns):>2d} cols")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
