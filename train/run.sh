#!/bin/bash
source ~/.bashrc

CONFIG_FILES=(
    "config_af3.py"                                 # 1. Change to your config file path
    "config_esmc_small.py"
    )

for CONFIG in "${CONFIG_FILES[@]}"; do
    echo "Running with config: $CONFIG"
    python ~/project/IMG/code/train.py $CONFIG      # 2. Change to your train.py path
    python ~/project/IMG/code/test.py $CONFIG       # 3. Change to your test.py path
done
