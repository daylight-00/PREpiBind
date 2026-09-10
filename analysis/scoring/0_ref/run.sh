#!/bin/bash
# Rescore the test sets with NetMHCIIpan-4.3 and MixMHC2pred-2.0, one process per allele.
#
#     NETMHCIIPAN_HOME=... MIXMHC2PRED_HOME=... ./run.sh
#
# Run prep.py first: it writes tasks_{net,mix}.txt and the per-allele inputs into the working
# directory. Neither tool uses a GPU, so this wants CPU cores, not an accelerator -- wrap it in
# whatever scheduler you have.
set -uo pipefail

PY=${PYTHON:-python}
J=${JOBS:-${SLURM_CPUS_PER_TASK:-$(nproc)}}

echo "host=$(hostname) cores=$J start=$(date -Is)"

# One process per allele, $J at a time, longest allele first (prep.py sorted the
# task list that way: with a fixed pool the wall clock is the longest task).
# --halt is deliberately not used - one allele failing should not discard the
# other 147; check.py reports what is missing and rerunning resumes, because
# score.py skips an output whose row count already matches its input.
for tool in net mix; do
    n=$(wc -l < "tasks_$tool.txt")
    echo "=== $tool: $n alleles on $J workers $(date -Is)"
    t0=$SECONDS
    xargs -a "tasks_$tool.txt" -P "$J" -n 1 -I{} \
        "$PY" score.py "$tool" {} > "log_$tool.txt" 2>&1
    rc=$?
    echo "$tool rc=$rc wall=$((SECONDS - t0))s scored=$(ls out/$tool | wc -l)/$n"
    grep -ciE 'error|traceback' "log_$tool.txt" | xargs -I{} echo "$tool error lines: {}"
done

echo "=== collect $(date -Is)"
"$PY" collect.py

echo "done=$(date -Is)"
