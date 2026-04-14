#!/bin/bash
git clone https://github.com/daylight-00/boltz
cd boltz
pip install -e .
cd ..

# Run inference for epitope sequences with MSA server
python ./boltz/src/boltz/main_test_click.py predict \
    input_boltz_hla \
    --out_dir output_boltz_hla \
    --use_msa_server

# Run inference for epitope sequences without MSA server
python ./boltz/src/boltz/main_test_click.py predict \
    input_boltz_epi \
    --out_dir output_boltz_epi \
    # --use_msa_server
