import os
import random
import re
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class DataProvider(Dataset):
    def __init__(
            self,
            epi_path, epi_args,
            hla_path, hla_args,
            shuffle=False,
            specific_hla=None,
            num_folds=None,
            ):
        self.epi_path = epi_path
        self.epi_args = epi_args
        self.hla_path = hla_path
        self.hla_args = hla_args
        self.shuffle = shuffle
        self.specific_hla = specific_hla
        self.num_folds = num_folds
        self.hla_seq_map = self.make_hla_seq_map()
        self.samples = self.get_samples()

    def normalize_hla_name(self, hla_name):
        hla_name = re.sub(r'\*|:|-', '', hla_name)
        return hla_name

    def make_hla_seq_map(self):
        hla_header = self.hla_args['hla_header']
        seq_header = self.hla_args['seq_header']
        seperator = self.hla_args['seperator']

        df_hla = pd.read_csv(self.hla_path, sep=seperator)
        df_hla = df_hla.dropna(subset=[hla_header, seq_header])
        # df_hla[hla_header] = df_hla[hla_header].apply(self.normalize_hla_name)
        hla_seq_map = dict(zip(df_hla[hla_header], df_hla[seq_header]))
        print(f'Number of HLA alleles: {len(hla_seq_map)}') if self.specific_hla is None else None
        return hla_seq_map

    def get_samples(self):
        hla_header = self.epi_args['hla_header']
        epi_header = self.epi_args['epi_header']
        fld_header = self.epi_args.get('fld_header', 'Fold')
        seperator = self.epi_args['seperator']
    
        df_epi = pd.read_csv(self.epi_path, sep=seperator)
        df_epi = df_epi.dropna(subset=[hla_header, epi_header])
        self.df_epi = df_epi.copy()
        # df_epi[hla_header] = df_epi[hla_header].apply(self.normalize_hla_name)
        # df_epi = df_epi[df_epi[hla_header].isin(self.hla_seq_map)]
        
        # self.top_10_hlas = df_epi['HLA_Name'].value_counts().nlargest(10).index.tolist()
        # if self.specific_hla is not None:
        #     df_epi = df_epi[df_epi[hla_header] == self.specific_hla]
        if self.num_folds is not None:
            self.fold_indices = [df_epi[df_epi[fld_header] == fold].index.tolist() for fold in range(self.num_folds)]

        samples = list(zip(df_epi[hla_header], df_epi[epi_header]))
        print(f'Number of samples: {len(samples)}') if self.specific_hla is None else None
        if self.shuffle:
            random.shuffle(samples)
        return samples
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        hla_name, epi_seq = self.samples[idx]
        if "_" in hla_name:
            hla_name_ = hla_name.split("_")
            hla_seq = (self.hla_seq_map[hla_name_[0]], self.hla_seq_map[hla_name_[1]])
        else:
            hla_seq = self.hla_seq_map[hla_name]
        return hla_name, epi_seq, hla_seq

class DataProvider_Cross(Dataset):
    def __init__(
            self,
            epi_path, epi_args,
            hla_path, hla_args,
            val_idx,
            shuffle=False,
            specific_hla=None
            ):
        self.epi_path = epi_path
        self.epi_args = epi_args
        self.hla_path = hla_path
        self.hla_args = hla_args
        self.shuffle = shuffle
        self.specific_hla = specific_hla

        self.hla_seq_map = self.make_hla_seq_map()
        self.samples = self.get_samples()

    def make_hla_seq_map(self):
        hla_header = self.hla_args['hla_header']
        seq_header = self.hla_args['seq_header']
        seperator = self.hla_args['seperator']

        df_hla = pd.read_csv(self.hla_path, sep=seperator)
        df_hla = df_hla.dropna()
        hla_seq_map = dict(zip(df_hla[hla_header], df_hla[seq_header]))
        print(f'Number of HLA alleles: {len(hla_seq_map)}') if self.specific_hla is None else None
        return hla_seq_map

    def get_samples(self):
        hla_header = self.epi_args['hla_header']
        epi_header = self.epi_args['epi_header']
        tgt_header = self.epi_args['tgt_header']
        fld_header = self.epi_args.get('fld_header', 'Fold')
        seperator = self.epi_args['seperator']

        df_epi = pd.read_csv(self.epi_path, sep=seperator)
        df_epi = df_epi.dropna()
        df_epi = df_epi[df_epi[hla_header].isin(self.hla_seq_map)]
        


        samples = list(zip(df_epi[hla_header], df_epi[epi_header], df_epi[tgt_header].astype(float)))
        print(f'Number of samples: {len(samples)}') if self.specific_hla is None else None
        if self.shuffle:
            random.shuffle(samples)
        return samples
    


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        hla_name, epi_seq, target = self.samples[idx]
        hla_seq = self.hla_seq_map[hla_name]
        return hla_name, epi_seq, target, hla_seq
