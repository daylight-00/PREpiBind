import pandas as pd
import os
from tqdm import tqdm
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from chai_lab.data.parsing.msas.aligned_pqt import merge_a3m_in_directory

databases = {
    "uniref90"      : "$AF3DB/uniref90_2022_05.fa",
    "uniprot"       : "$AF3DB/uniprot_all_2021_04.fa",
    "bfd_uniclust"  : "$AF3DB/bfd-first_non_consensus_sequences.fasta",
    "mgnify"        : "$AF3DB/mgy_clusters_2022_05.fa"
}

def run_jackhmmer(db_name,h, input_sequence, output_dir):
    output_txt = os.path.join(output_dir, f"hits_{db_name}.txt")
    output_sto = os.path.join(output_dir, f"hits_{db_name}.sto")
    output_a3m = os.path.join(output_dir, f"hits_{db_name}.a3m")
    jackhmmer_cmd = [
        "$HMMER/hmmer/bin/jackhmmer",
        "-o", output_txt,
        "-A", output_sto,
        "-N", "1",
        "-E", "0.0001",
        "--incE", "0.0001",
        "--F1", "0.0005",
        "--F2", "0.00005",
        "--F3", "0.0000005",
        "--cpu", "8",
        input_sequence,
        db_path
    ]
    reformat_cmd = [
        "reformat.pl",
        "sto",
        "a3m",
        output_sto,
        output_a3m
    ]
    try:
        subprocess.run(jackhmmer_cmd, check=True)
        subprocess.run(reformat_cmd, check=True)
        merge_a3m_in_directory(output_dir)
        if not os.path.exists(output_sto):
            return f"Failed {db_name} for {input_sequence}: STO file not created."
        return f"Completed {db_name} for {input_sequence}"
    except subprocess.CalledProcessError as e:
        return f"Failed {db_name} for {input_sequence}: {str(e)}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

input_path = '../../data/mhc_mapping/mhc_mapping_light.csv'
df = pd.read_csv(input_path)
input_dir = 'input'
output_base_dir = 'output'
os.makedirs(input_dir, exist_ok=True)
os.makedirs(output_base_dir, exist_ok=True)

TOTAL_CPU = 104*4
MAX_WORKERS = TOTAL_CPU // 8

tasks = []

for index, row in tqdm(df.iterrows(), total=len(df)):
    name, sequence = row['HLA_Name'], row['HLA_Seq']
    with open(os.path.join(input_dir, f"{name}.fasta"), 'w') as f:
        f.write(f">{name}\n{sequence}")
    
    input_sequence = os.path.join(input_dir, f"{name}.fasta")
    output_dir = os.path.join(output_base_dir, name)
    os.makedirs(output_dir, exist_ok=True)
    
    for db_name, db_path in databases.items():
        tasks.append((db_name, db_path, input_sequence, output_dir))

results = []
with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
    future_to_task = {executor.submit(run_jackhmmer, *task): task for task in tasks}
    
    for future in tqdm(as_completed(future_to_task), total=len(future_to_task)):
        try:
            result = future.result()
            results.append(result)
        except Exception as e:
            results.append(f"Task failed with error: {str(e)}")

for result in results:
    print(result)
