#!/bin/bash
git clone https://github.com/daylight-00/chai-lab
cd chai-lab
pip install -e .
cd ..

# Prepare A3M files for HLA sequences
python chai_make_a3m.py

# Run inference for HLA sequences
python chai_hla.py

# Run inference for epitope sequences
python chai_epi.py
