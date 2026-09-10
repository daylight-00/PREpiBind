import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

def pad_and_mask_collate_fn(batch):
    """Pad a batch of embeddings and build the padding masks.

    Args:
        batch: list of (hla_emb_s, hla_emb_p, epi_emb_s, epi_emb_p, target) tuples from the
            Dataset. The pair entries hla_emb_p and epi_emb_p are either a Tensor for every
            item in the batch or False for every item; mixed batches are not supported.

    Returns:
        (padded_hla_s, padded_hla_p, padded_epi_s, padded_epi_p, mask_hla, mask_epi, targets),
        where the pair tensors are zero placeholders when the representation has no pair channel
        and the masks are True at padded positions.
    """
    hla_s_list, hla_p_list, epi_s_list, epi_p_list, target_list = zip(*batch)
    batch_size = len(target_list)

    # HLA single-residue embeddings
    hla_s_lens = [len(emb) for emb in hla_s_list]
    max_hla_len = max(hla_s_lens) if batch_size > 0 and hla_s_lens else 0
    padded_hla_s = pad_sequence(hla_s_list, batch_first=True, padding_value=0.0)
    mask_hla = torch.ones(batch_size, max_hla_len, dtype=torch.bool)
    for i, length in enumerate(hla_s_lens):
        mask_hla[i, :length] = False

    # Epitope single-residue embeddings
    epi_s_lens = [len(emb) for emb in epi_s_list]
    max_epi_len = max(epi_s_lens) if batch_size > 0 and epi_s_lens else 0
    padded_epi_s = pad_sequence(epi_s_list, batch_first=True, padding_value=0.0)
    mask_epi = torch.ones(batch_size, max_epi_len, dtype=torch.bool)
    for i, length in enumerate(epi_s_lens):
        mask_epi[i, :length] = False

    # HLA pair-derived embeddings, or a zero placeholder when the representation has none
    if batch_size > 0 and isinstance(hla_p_list[0], torch.Tensor):
        padded_hla_p = pad_sequence(hla_p_list, batch_first=True, padding_value=0.0)
        # match the single-embedding length
        current_max_p_len = padded_hla_p.shape[1]
        target_len = max_hla_len
        hla_p_dim = padded_hla_p.shape[2] if padded_hla_p.ndim == 3 else 128
        if current_max_p_len < target_len:
             pad_width = target_len - current_max_p_len
             padded_hla_p = padded_hla_p.permute(0, 2, 1)
             padded_hla_p = nn.functional.pad(padded_hla_p, (0, pad_width))
             padded_hla_p = padded_hla_p.permute(0, 2, 1)
        elif current_max_p_len > target_len:
             padded_hla_p = padded_hla_p[:, :target_len, :]
    else:  # no pair channel for this representation
        hla_p_dim = 128
        padded_hla_p = torch.zeros(batch_size, max_hla_len, hla_p_dim, dtype=torch.float32)

    # Epitope pair-derived embeddings, same convention
    if batch_size > 0 and isinstance(epi_p_list[0], torch.Tensor):
        padded_epi_p = pad_sequence(epi_p_list, batch_first=True, padding_value=0.0)
        # match the single-embedding length
        current_max_p_len = padded_epi_p.shape[1]
        target_len = max_epi_len
        epi_p_dim = padded_epi_p.shape[2] if padded_epi_p.ndim == 3 else 128
        if current_max_p_len < target_len:
             pad_width = target_len - current_max_p_len
             padded_epi_p = padded_epi_p.permute(0, 2, 1)
             padded_epi_p = nn.functional.pad(padded_epi_p, (0, pad_width))
             padded_epi_p = padded_epi_p.permute(0, 2, 1)
        elif current_max_p_len > target_len:
             padded_epi_p = padded_epi_p[:, :target_len, :]
    else:  # no pair channel for this representation
        epi_p_dim = 128
        padded_epi_p = torch.zeros(batch_size, max_epi_len, epi_p_dim, dtype=torch.float32)

    # Targets
    targets = torch.stack(target_list)

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

_tokenizer = None


def _esm_tokenizer():
    """Only the inference path tokenises, so the tokenizer is built on first use. Since vendoring
    it needs nothing beyond torch; training and analysis read precomputed embeddings and still run
    without esm or flash-attn installed."""
    global _tokenizer
    if _tokenizer is None:
        from prepibind.esmc import EsmSequenceTokenizer
        _tokenizer = EsmSequenceTokenizer()
    return _tokenizer


def pad_and_mask_collate_fn_inf(batch):
    """batch: list of (hla_emb, epi_seq)"""
    tokenizer = _esm_tokenizer()
    # Unpack batch
    hla_list, epi_list = zip(*batch)
    batch_size = len(epi_list)

    # Pad HLA embeddings
    hla_s_lens = [len(emb) for emb in hla_list]
    max_hla_len = max(hla_s_lens) if batch_size > 0 else 0
    # float32, as in training: encoder.py upcasts the float16 store before the head sees it.
    padded_hla = pad_sequence(hla_list, batch_first=True, padding_value=0.0).to(torch.float32)
    mask_hla = torch.ones(batch_size, max_hla_len, dtype=torch.bool)
    for i, length in enumerate(hla_s_lens):
        mask_hla[i, :length] = False

    # Pad epitope token sequences
    pad_token_id = tokenizer.pad_token_id
    max_epi_len = max(len(x) for x in epi_list)
    epi_tensor = torch.full((batch_size, max_epi_len), pad_token_id, dtype=torch.long)
    mask_epi = torch.ones(batch_size, max_epi_len, dtype=torch.bool)
    for i, tks in enumerate(epi_list):
        tks_tensor = torch.as_tensor(tks, dtype=torch.long)
        epi_tensor[i, :len(tks_tensor)] = tks_tensor
        mask_epi[i, :len(tks_tensor)] = False
    mask_epi = mask_epi[:, 1:-1]  # Exclude [CLS] and [SEP] tokens

    return (
        padded_hla,    # (B, max_hla_len, D_hla)
        epi_tensor,    # (B, max_epi_len) long
        mask_hla,      # (B, max_hla_len) bool
        mask_epi       # (B, max_epi_len-2) bool
    )
