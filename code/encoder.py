import torch
import numpy as np
from torch.utils.data import Dataset
import h5py
import time
from esm.tokenization import EsmSequenceTokenizer
from esm.utils import encoding

#%% PLM
def get_plm_emb(emb_dict, key, start_idx_a=None, end_idx_a=None, max_retries=5, retry_delay=0.1):
    for attempt in range(max_retries):
        try:
            embedding = np.squeeze(emb_dict[key][()])
            embedding = torch.tensor(embedding, dtype=torch.float32)
            if start_idx_a is not None and end_idx_a is not None:
                embedding = embedding[start_idx_a:end_idx_a]
            return embedding
        except OSError as e:
            print(f"[get_plm_emb] OSError occured (Attemp {attempt + 1}/{max_retries}) — key: {key}")
            print(f"Error message: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise

class plm_plm_mask_msa_pair_inf(Dataset):
    def __init__(self, data_provider, hla_emb_path):
        self.data_provider = data_provider
        self.hdf5_path = hla_emb_path
        self.hla_emb_dict = h5py.File(self.hdf5_path, 'r', libver='latest')
        self.tokenizer = EsmSequenceTokenizer()

    def __len__(self):
        return len(self.data_provider)

    def __getitem__(self, idx):
        hla_name, epi_seq, _ = self.data_provider[idx]
        hla_name_a, hla_name_b = hla_name.split("_")
        hla_emb_a = get_plm_emb(self.hla_emb_dict, hla_name_a)
        hla_emb_b = get_plm_emb(self.hla_emb_dict, hla_name_b)
        hla_emb = torch.cat([hla_emb_a, hla_emb_b], dim=0)
        epi_seq = encoding.tokenize_sequence(epi_seq, self.tokenizer, add_special_tokens=True)
        return hla_emb, epi_seq

    def __del__(self):
        self.hla_emb_dict.close()
