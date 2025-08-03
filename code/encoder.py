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

def split_hla(hla_seq):
    if "|" in hla_seq:
        hla_seq, start_idx, end_idx = hla_seq.split('|')
        start_idx = int(start_idx)
        end_idx = int(end_idx)
        hla_seq = hla_seq[start_idx:end_idx]
    else:
        start_idx = None
        end_idx = None
    return hla_seq, start_idx, end_idx

class plm_plm_mask_msa_pair(Dataset):
    def __init__(self, data_provider, hla_emb_path_s, epi_emb_path_s, hla_emb_path_p=None, epi_emb_path_p=None):
        self.data_provider = data_provider
        self.hdf5_path_s1 = hla_emb_path_s
        self.hdf5_path_s2 = epi_emb_path_s
        self.hdf5_path_p1 = hla_emb_path_p
        self.hdf5_path_p2 = epi_emb_path_p
        self.hla_emb_dict_s = h5py.File(self.hdf5_path_s1, 'r', libver='latest')
        self.epi_emb_dict_s = h5py.File(self.hdf5_path_s2, 'r', libver='latest')
        self.hla_emb_dict_p = h5py.File(self.hdf5_path_p1, 'r', libver='latest') if hla_emb_path_p is not None else None
        self.epi_emb_dict_p = h5py.File(self.hdf5_path_p2, 'r', libver='latest') if epi_emb_path_p is not None else None

    def __len__(self):
        return len(self.data_provider)

    def __getitem__(self, idx):
        hla_name, epi_seq, target, hla_seq = self.data_provider[idx]
        if "_" in hla_name:
            hla_seq_a, hla_seq_b = hla_seq
            hla_name_a, hla_name_b = hla_name.split("_")
            hla_seq_a, start_idx_a, end_idx_a = split_hla(hla_seq_a)
            hla_seq_b, start_idx_b, end_idx_b = split_hla(hla_seq_b)
            hla_emb_s_a = get_plm_emb(self.hla_emb_dict_s, hla_name_a, start_idx_a, end_idx_a)
            hla_emb_s_b = get_plm_emb(self.hla_emb_dict_s, hla_name_b, start_idx_b, end_idx_b)
            hla_emb_s = torch.cat([hla_emb_s_a, hla_emb_s_b], dim=0)
            hla_emb_p_a = get_plm_emb(self.hla_emb_dict_p, hla_name_a, start_idx_a, end_idx_a) if self.hla_emb_dict_p is not None else False
            hla_emb_p_b = get_plm_emb(self.hla_emb_dict_p, hla_name_b, start_idx_b, end_idx_b) if self.hla_emb_dict_p is not None else False
            hla_emb_p = torch.cat([hla_emb_p_a, hla_emb_p_b], dim=0) if self.hla_emb_dict_p is not None else False
        else:
            hla_seq, start_idx, end_idx = split_hla(hla_seq)
            hla_emb_s = get_plm_emb(self.hla_emb_dict_s, hla_name, start_idx, end_idx)
            hla_emb_p = get_plm_emb(self.hla_emb_dict_p, hla_name, start_idx, end_idx) if self.hla_emb_dict_p is not None else False
        # Load embeddings
        epi_emb_s = get_plm_emb(self.epi_emb_dict_s, epi_seq)
        epi_emb_p = get_plm_emb(self.epi_emb_dict_p, epi_seq) if self.epi_emb_dict_p is not None else False
        # Convert target to tensor
        target = torch.tensor(target, dtype=torch.float32).unsqueeze(0)
        return hla_emb_s, hla_emb_p, epi_emb_s, epi_emb_p, target

    def __del__(self):
        self.hla_emb_dict_s.close()
        self.epi_emb_dict_s.close()
        self.hla_emb_dict_p.close() if self.hla_emb_dict_p is not None else None
        self.epi_emb_dict_p.close() if self.epi_emb_dict_p is not None else None

#%% BLOSUM
def get_blosum_emb(matrix, sequence, start_idx=None, end_idx=None):
    embedding = [matrix[aa] for aa in sequence]
    embedding = torch.tensor(embedding, dtype=torch.float32)
    if start_idx is not None and end_idx is not None:
        embedding = embedding[start_idx:end_idx]
    return embedding

class blosum_mask_msa_pair(Dataset):
    def __init__(self, data_provider, hla_emb_path_s=None, epi_emb_path_s=None, hla_emb_path_p=None, epi_emb_path_p=None):
        self.data_provider = data_provider
        from utils.matrix import blosum62
        self.blosum62 = blosum62

    def __len__(self):
        return len(self.data_provider)

    def __getitem__(self, idx):
        hla_name, epi_seq, target, hla_seq = self.data_provider[idx]
        if "_" in hla_name:
            hla_seq_a, hla_seq_b = hla_seq
            hla_name_a, hla_name_b = hla_name.split("_")
            hla_seq_a, start_idx_a, end_idx_a = split_hla(hla_seq_a)
            hla_seq_b, start_idx_b, end_idx_b = split_hla(hla_seq_b)
            hla_emb_s_a = get_blosum_emb(self.blosum62, hla_seq_a, start_idx_a, end_idx_a)
            hla_emb_s_b = get_blosum_emb(self.blosum62, hla_seq_b, start_idx_b, end_idx_b)
            hla_emb_s = torch.cat([hla_emb_s_a, hla_emb_s_b], dim=0)
        else:
            hla_seq, start_idx, end_idx = split_hla
            hla_emb_s = get_blosum_emb(self.blosum62, hla_seq, start_idx, end_idx)
        epi_emb_s = get_blosum_emb(self.blosum62, epi_seq)
        hla_emb_p = False
        epi_emb_p = False
        # Convert target to tensor
        target = torch.tensor(target, dtype=torch.float32).unsqueeze(0)
        return hla_emb_s, hla_emb_p, epi_emb_s, epi_emb_p, target

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

#%% DEEPNEO
class deepneo(Dataset):
    def __init__(self, data_provider, matrix_size=(15, 269)):
        self.data_provider = data_provider
        self.matrix_size = matrix_size
        from utils.matrix import get_calpha_matrix
        self.get_calpha_matrix = get_calpha_matrix

    def __len__(self):
        return len(self.data_provider)

    def __getitem__(self, idx):
        hla_name, epi_seq, target, hla_seq = self.data_provider[idx]
        encoded_matrix = self.get_calpha_matrix(hla_seq, epi_seq, self.matrix_size)
        target = torch.tensor(target, dtype=torch.float32).unsqueeze(0)
        encoded_matrix = torch.tensor(encoded_matrix, dtype=torch.float32).unsqueeze(0)
        return encoded_matrix, target
