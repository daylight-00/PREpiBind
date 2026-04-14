import torch
import pandas as pd
from esm.sdk.forge import ESM3ForgeInferenceClient
from esm.sdk.api import ESMProtein, LogitsConfig
import h5py
from tqdm import tqdm
from multiprocessing import Pool, Lock, cpu_count
import time

input_path = '../../data/mhc_mapping/HLA2_IMGT_light.csv'
# input_path = '../../data/unique_epitope_whole.csv'
data = pd.read_csv(input_path)
output_path = 'emb_hla_esm3_small.h5'
# output_path = 'emb_epi_esm3_small.h5'

file_lock = Lock()

def init_client():
    global client
    client = ESM3ForgeInferenceClient(
        model="esm3-small-2024-08",
        url="https://forge.evolutionaryscale.ai",
        token=YOUR_API_TOKEN_HERE
    )

def process_row(row):
    name, sequence = row['HLA_Name'], row['HLA_Seq']
    # name, sequence = row['Epi_Seq'], row['Epi_Seq']
    protein = ESMProtein(sequence=sequence, potential_sequence_of_concern=True)
    while True:
        try:
            protein_tensor = client.encode(protein)
            logits_output = client.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True))
            embedding = logits_output.embeddings.cpu().detach().squeeze(0).numpy()[1:-1,:]
            break
        except Exception as e:
            # print(f"Error processing sequence {name}: {e}. Retrying in 10 seconds...")
            time.sleep(10)
    with file_lock:
        with h5py.File(output_path, 'a') as h5file:
            h5file.create_dataset(name, data=embedding)

if __name__ == '__main__':
    with Pool(processes=cpu_count(), initializer=init_client) as pool:
        list(tqdm(pool.imap_unordered(process_row, [row for _, row in data.iterrows()]), total=len(data), desc="Processing sequences"))
    print(f"Embeddings have been saved to {output_path}")
