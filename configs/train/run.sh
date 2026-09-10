#!/bin/bash
# Train and evaluate one representation, or several in sequence.
#
#     ./run.sh config_esmc_small.py                 # one
#     ./run.sh config_*.py                          # all eleven
#
# Each config declares its model, encoder, embedding stores and learning rate; the shared settings
# are in config_global.json beside them. The embedding stores are not distributed with the
# repository -- set PREPIBIND_EMB_ROOT, or see pipeline/embeddings/.
set -euo pipefail

if [ $# -eq 0 ]; then
    echo "usage: $0 <config.py> [config.py ...]" >&2
    exit 2
fi

for config in "$@"; do
    echo "=== $config"
    python -m prepibind.train "$config"
    python -m prepibind.test  "$config"
done
