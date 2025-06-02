import torch
from torch.nn.utils.rnn import pad_sequence
from esm.utils import encoding
from esm.tokenization import EsmSequenceTokenizer
tokenizer = EsmSequenceTokenizer()

def pad_and_mask_collate_fn_inf(batch):
    """
    batch: list of (hla_emb_s, epi_seq, target)
    """
    # 데이터 분리
    hla_s_list, epi_s_list = zip(*batch)
    batch_size = len(epi_s_list)
    
    # HLA_s 처리
    hla_s_lens = [len(emb) for emb in hla_s_list]
    max_hla_len = max(hla_s_lens) if batch_size > 0 else 0
    padded_hla_s = pad_sequence(hla_s_list, batch_first=True, padding_value=0.0)
    mask_hla = torch.ones(batch_size, max_hla_len, dtype=torch.bool)
    for i, length in enumerate(hla_s_lens):
        mask_hla[i, :length] = False

    # epi_s_list 토큰화 + 패딩
    pad_token_id = tokenizer.pad_token_id
    max_epi_len = max(len(x) for x in epi_s_list)
    epi_s_tensor = torch.full((batch_size, max_epi_len), pad_token_id, dtype=torch.long)
    mask_epi = torch.ones(batch_size, max_epi_len, dtype=torch.bool)
    for i, tks in enumerate(epi_s_list):
        tks_tensor = torch.as_tensor(tks, dtype=torch.long)
        epi_s_tensor[i, :len(tks_tensor)] = tks_tensor
        mask_epi[i, :len(tks_tensor)] = False   # 패딩 아닌 부분은 False
    mask_epi = mask_epi[:, 1:-1] # [CLS]와 [SEP] 토큰 제외

    return (
        padded_hla_s,    # (B, max_hla_len, D_hla)
        mask_hla,        # (B, max_hla_len) bool
        epi_s_tensor,    # (B, max_epi_len) long
        mask_epi         # (B, max_epi_len) bool
    )
