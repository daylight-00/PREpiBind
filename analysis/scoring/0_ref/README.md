# 0_ref - NetMHCIIpan-4.3 / MixMHC2pred-2.0 predictions

This stage produces *predictions*, not metrics. The metrics are computed by
`pipeline.ref_level()` / `pipeline.ref_allele_level()` as part of every analysis,
so there is exactly one place that decides which rows a number was measured on.

    prep.py       union of every test set's (peptide, allele) pairs -> allele .pep files
    score.py      one tool x one allele = one task
    run.sh        xargs, one process per allele, $JOBS at a time, longest allele first
    collect.py    out/ -> pred_netmhcpan.csv, pred_mixmhcpred.csv, coverage.csv
    verify.py     batching is a no-op / old numbers reproduce / nothing missing
    allelemap.py  HLA_Name -> each tool's allele name, checked against 125 known-good

Run it in scratch, following the `0_raw` mirror convention. `run.sh` carries no
scheduler directives - wrap it in whatever your site uses; `JOBS` falls back to
`$SLURM_CPUS_PER_TASK`, then `nproc`:

    mkdir -p $SCRATCH/260830/ref
    cp *.py run.sh $SCRATCH/260830/ref/
    cd $SCRATCH/260830/ref && python prep.py
    NETMHCIIPAN_HOME=... MIXMHC2PRED_HOME=... JOBS=112 ./run.sh
    python verify.py

Then snapshot the two prediction tables to `0_raw/260830/ref/`, which is where
`pipeline.REF_PRED` reads them from via `rp.at()`. `verify.py` reads the snapshot
too, so its first two checks run from here; the coverage check reads
`coverage.csv`, which `collect.py` leaves in the working directory and which is
not shipped.

Kept here for provenance: `allele_map.csv` (all 149 alleles, each tool's name and
whether the tool supports it).

Tool versions: NetMHCIIpan **4.3g**, MixMHC2pred **2.0.2**, both from
`$SCRATCH/250604/1_ref/`. Neither uses a GPU.
