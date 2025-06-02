import torch
from torch.nn.utils.rnn import pad_sequence
# Dataset 클래스 및 get_plm_emb 함수는 이전에 제공된 것을 사용한다고 가정합니다.
# (Dataset은 hla_emb_p/epi_emb_p에 False를 반환할 수 있음)

def pad_and_mask_collate_fn_inf(batch):
    """
    batch: list of (hla_emb_s, epi_seq, target)
    - hla_emb_s: Tensor (L_i, D)
    - epi_seq: string (동일 길이)
    - target: Tensor 또는 숫자
    """
    # 1. 데이터 분리
    hla_s_list, epi_s_list, target_list = zip(*batch)
    batch_size = len(target_list)
    
    # --- HLA_s 처리 ---
    hla_s_lens = [len(emb) for emb in hla_s_list]
    max_hla_len = max(hla_s_lens) if batch_size > 0 else 0
    padded_hla_s = pad_sequence(hla_s_list, batch_first=True, padding_value=0.0)
    mask_hla = torch.ones(batch_size, max_hla_len, dtype=torch.bool)
    for i, length in enumerate(hla_s_lens):
        mask_hla[i, :length] = False

    # --- epi_s_list는 string 리스트 그대로 사용 ---
    # epi_s_list: (batch, str) 동일 길이

    # --- 타겟 처리 ---
    if isinstance(target_list[0], torch.Tensor):
        targets = torch.stack(target_list)
    else:
        targets = torch.tensor(target_list)

    # --- 최종 반환 ---
    return (
        padded_hla_s,    # (B, max_hla_len, D_hla)
        mask_hla,        # (B, max_hla_len) bool
        epi_s_list,      # (B,) string list
        targets          # (B, 1) or (B,)
    )
