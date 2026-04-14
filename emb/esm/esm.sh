#!/bin/bash
pip install esm

# Run inference for HLA sequences with ESM3 Small
python esm_api_esm3_small_2408.py

# Run inference for HLA sequences with ESM C 300M
python esm_local_esmc_small.py
