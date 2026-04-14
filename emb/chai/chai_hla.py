from chai_lab.chai1_custom import run_inference
from pathlib import Path
from tqdm import tqdm
import torch
import tempfile
import pandas as pd
import h5py

input_path = '../../data/mhc_mapping/HLA2_IMGT_light.csv'
# input_path = '../../data/unique_epitope_whole.csv'
data = pd.read_csv(input_path)
total_sequences = len(data)

for idx, row in tqdm(data.iterrows(), desc="Processing sequences", total=total_sequences):
    name, sequence = row['HLA_Name'], row['HLA_Seq']
    # name, sequence = row['Epi_Seq'], row['Epi_Seq']
    fasta = f"""\n>protein|name={name}\n{sequence}\n""".strip()
    with tempfile.NamedTemporaryFile(delete=True, mode='w', suffix=".fasta") as fasta_tmp:
        fasta_tmp.write(fasta)
        fasta_tmp.flush()
        trunk_sing, trunk_pair = run_inference(
            fasta_file=Path(fasta_tmp.name),
            output_dir=Path("output_chai_jack")/name,
            seed=42,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            use_esm_embeddings=True,
            # use_msa_server=True,
            msa_directory=Path("./output")/name
        )
    trunk_sing = trunk_sing.squeeze(0)
    trunk_pair = trunk_pair.squeeze(0)
    trunk_sing=trunk_sing[:len(sequence)]
    trunk_pair=trunk_pair[:len(sequence),:len(sequence),:]
    with h5py.File(f"../emb_hla_chai_single.h5", "a") as f1, h5py.File(f"../emb_hla_chai_pair.h5", "a") as f2:
        f1.create_dataset(name, data=trunk_sing.to(torch.float32).cpu().numpy())
        f2.create_dataset(name, data=trunk_pair.to(torch.float32).cpu().numpy())
