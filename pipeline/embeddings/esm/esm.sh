#!/bin/bash
# environment: see pyproject.toml / pixi.lock in this directory

# Run inference for HLA sequences with ESM3 Small
python esm_api_esm3_small_2408.py

# Run inference for HLA sequences with ESM C 300M
python esm_local_esmc_300m.py
