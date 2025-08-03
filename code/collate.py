import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

def pad_and_mask_collate_fn(batch):
    """
    Dataset에서 반환된 임베딩 배치에 대해 패딩과 마스크를 생성하는 collate_fn.
    유효성 검사/필터링 로직을 제거하여 간소화된 버전입니다.
    입력 데이터가 올바르다고 가정하며, 문제가 있으면 표준 오류가 발생합니다.

    Args:
        batch: Dataset의 __getitem__에서 반환된 튜플들의 리스트.
               각 튜플: (hla_emb_s, hla_emb_p, epi_emb_s, epi_emb_p, target)
               * hla_emb_p, epi_emb_p는 Tensor 또는 False일 수 있음 (배치 내 동질적 가정).

    Returns:
        튜플 형태의 배치 데이터:
        (
            padded_hla_s,    # Tensor (B, max_hla_len, D_hla)
            padded_hla_p,    # Tensor (B, max_hla_len, D_hla) - 없으면 0벡터 Placeholder
            padded_epi_s,    # Tensor (B, max_epi_len, D_epi)
            padded_epi_p,    # Tensor (B, max_epi_len, D_epi) - 없으면 0벡터 Placeholder
            mask_hla,        # Tensor (B, max_hla_len), bool
            mask_epi,        # Tensor (B, max_epi_len), bool
            targets          # Tensor (B, 1)
        )
    """
    # 1. 데이터 분리 (오류 발생 가능: batch가 비어있거나 형식이 다르면)
    hla_s_list, hla_p_list, epi_s_list, epi_p_list, target_list = zip(*batch)
    batch_size = len(target_list)

    # --- HLA_s 처리 (hla_s_list가 Tensor 리스트라고 가정) ---
    # 오류 발생 가능: 리스트에 Tensor가 아닌 객체가 있거나, len() 계산 불가 시
    hla_s_lens = [len(emb) for emb in hla_s_list]
    max_hla_len = max(hla_s_lens) if batch_size > 0 and hla_s_lens else 0
    # 오류 발생 가능: pad_sequence에 Tensor가 아닌 객체 전달 시
    padded_hla_s = pad_sequence(hla_s_list, batch_first=True, padding_value=0.0)
    mask_hla = torch.ones(batch_size, max_hla_len, dtype=torch.bool)
    for i, length in enumerate(hla_s_lens):
        mask_hla[i, :length] = False

    # --- Epi_s 처리 (epi_s_list가 Tensor 리스트라고 가정) ---
    epi_s_lens = [len(emb) for emb in epi_s_list]
    max_epi_len = max(epi_s_lens) if batch_size > 0 and epi_s_lens else 0
    padded_epi_s = pad_sequence(epi_s_list, batch_first=True, padding_value=0.0)
    mask_epi = torch.ones(batch_size, max_epi_len, dtype=torch.bool)
    for i, length in enumerate(epi_s_lens):
        mask_epi[i, :length] = False

    # --- HLA_p 처리 (리스트가 모두 Tensor 이거나 모두 False 라고 가정) ---
    # 오류 발생 가능: 리스트가 비어있거나 첫 요소 접근 불가 시
    if batch_size > 0 and isinstance(hla_p_list[0], torch.Tensor):
        # 오류 발생 가능: pad_sequence에 Tensor가 아닌 객체 전달 시 (가정에 의해 발생 안 함)
        padded_hla_p = pad_sequence(hla_p_list, batch_first=True, padding_value=0.0)
        # 길이 맞추기
        current_max_p_len = padded_hla_p.shape[1]
        target_len = max_hla_len
        # 오류 발생 가능: shape 접근 불가 시
        hla_p_dim = padded_hla_p.shape[2] if padded_hla_p.ndim == 3 else 128 # 차원 확인 및 기본값
        if current_max_p_len < target_len:
             pad_width = target_len - current_max_p_len
             padded_hla_p = padded_hla_p.permute(0, 2, 1)
             padded_hla_p = nn.functional.pad(padded_hla_p, (0, pad_width))
             padded_hla_p = padded_hla_p.permute(0, 2, 1)
        elif current_max_p_len > target_len:
             padded_hla_p = padded_hla_p[:, :target_len, :]
    else: # 모두 False 또는 None 이라고 가정
        hla_p_dim = 128 # 기본값 또는 config에서 가져오기 (차원 정보 필요)
        padded_hla_p = torch.zeros(batch_size, max_hla_len, hla_p_dim, dtype=torch.float32)

    # --- Epi_p 처리 (리스트가 모두 Tensor 이거나 모두 False 라고 가정) ---
    if batch_size > 0 and isinstance(epi_p_list[0], torch.Tensor):
        padded_epi_p = pad_sequence(epi_p_list, batch_first=True, padding_value=0.0)
        # 길이 맞추기
        current_max_p_len = padded_epi_p.shape[1]
        target_len = max_epi_len
        epi_p_dim = padded_epi_p.shape[2] if padded_epi_p.ndim == 3 else 128 # 차원 확인 및 기본값
        if current_max_p_len < target_len:
             pad_width = target_len - current_max_p_len
             padded_epi_p = padded_epi_p.permute(0, 2, 1)
             padded_epi_p = nn.functional.pad(padded_epi_p, (0, pad_width))
             padded_epi_p = padded_epi_p.permute(0, 2, 1)
        elif current_max_p_len > target_len:
             padded_epi_p = padded_epi_p[:, :target_len, :]
    else: # 모두 False 또는 None 이라고 가정
        epi_p_dim = 128 # 기본값 또는 config에서 가져오기
        padded_epi_p = torch.zeros(batch_size, max_epi_len, epi_p_dim, dtype=torch.float32)

    # --- 타겟 처리 ---
    # 오류 발생 가능: target_list의 요소가 Tensor가 아니거나 stack 불가 시
    targets = torch.stack(target_list)
    # print(padded_hla_s.shape, padded_hla_p.shape)
    # print(mask_hla.sum(dim=0))

    # --- 최종 튜플 반환 ---
    return (
        padded_hla_s,
        padded_hla_p,
        padded_epi_s,
        padded_epi_p,
        mask_hla,
        mask_epi,
        targets
    )

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
    hla_list, epi_list = zip(*batch)
    batch_size = len(epi_list)
    
    # HLA_s 처리
    hla_s_lens = [len(emb) for emb in hla_list]
    max_hla_len = max(hla_s_lens) if batch_size > 0 else 0
    padded_hla = pad_sequence(hla_list, batch_first=True, padding_value=0.0).to(torch.float16)
    mask_hla = torch.ones(batch_size, max_hla_len, dtype=torch.bool)
    for i, length in enumerate(hla_s_lens):
        mask_hla[i, :length] = False

    # epi_list 토큰화 + 패딩
    pad_token_id = tokenizer.pad_token_id
    max_epi_len = max(len(x) for x in epi_list)
    epi_tensor = torch.full((batch_size, max_epi_len), pad_token_id, dtype=torch.long)
    mask_epi = torch.ones(batch_size, max_epi_len, dtype=torch.bool)
    for i, tks in enumerate(epi_list):
        tks_tensor = torch.as_tensor(tks, dtype=torch.long)
        epi_tensor[i, :len(tks_tensor)] = tks_tensor
        mask_epi[i, :len(tks_tensor)] = False   # 패딩 아닌 부분은 False
    mask_epi = mask_epi[:, 1:-1] # [CLS]와 [SEP] 토큰 제외

    return (
        padded_hla,    # (B, max_hla_len, D_hla)
        epi_tensor,    # (B, max_epi_len) long
        mask_hla,        # (B, max_hla_len) bool
        mask_epi         # (B, max_epi_len) bool
    )
