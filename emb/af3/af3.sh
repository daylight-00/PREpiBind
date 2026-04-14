#!/bin/bash
git clone https://github.com/daylight-00/alphafold3
cd alphafold3
pip install -r dev-requirements.txt
pip install --no-deps .
build_data
XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
XLA_PYTHON_CLIENT_PREALLOCATE=true
XLA_CLIENT_MEM_FRACTION=0.95
cd ..

# Prepare MSA for HLA sequences
python ./alphafold3/run_alphafold_custom.py \
    --run_inference=false \
    --run_data_pipeline=true \
    --jackhmmer_n_cpu 8 \
    --input_dir "json_hla/$1" \
    --output_dir "json_msa" \

# Run inference for HLA sequences with prepared MSA
python ./alphafold3/run_alphafold_custom.py \
    --run_inference=true \
    --run_data_pipeline=false \
    --jackhmmer_n_cpu 8 \
    --input_dir "input_af3_hla" \
    --output_dir "output_af3_hla" \

# Run inference for epitope sequences without MSA
python ./alphafold3/run_alphafold_custom.py \
    --run_inference=true \
    --run_data_pipeline=false \
    --jackhmmer_n_cpu 8 \
    --input_dir "input_af3_epi" \
    --output_dir "output_af3_epi" \
