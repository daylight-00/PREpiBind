import torch
import pandas as pd
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
import h5py
from tqdm import tqdm

input_path = '../../data/mhc_mapping/mhc_mapping_light.csv'
# input_path = '../../data/unique_epitope_whole.csv'
data = pd.read_csv(input_path)
total_sequences = len(data)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
client = ESMC.from_pretrained("esmc_300m").to("cuda")

output_path = '../emb_hla_esmc_300m.h5'
# output_path = '../emb_epi_esmc_300m.h5'
for idx, row in tqdm(data.iterrows(), desc="Processing sequences", total=total_sequences):
    name, sequence = row['HLA_Name'], row['HLA_Seq']
    # name, sequence = row['Epi_Seq'], row['Epi_Seq']
    protein = ESMProtein(sequence=sequence)
    protein_tensor = client.encode(protein)
    logits_output = client.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True))
    embedding = logits_output.embeddings.cpu().detach().squeeze(0).numpy()[1:-1,:]
    with h5py.File(output_path, 'a') as h5file:
        h5file.create_dataset(name, data=embedding)
print(f"Embeddings have been saved to {output_path}")
