# Vendored from esm 3.4.0: esm/tokenization/sequence_tokenizer.py, esm/utils/constants/esm3.py
# (SEQUENCE_VOCAB, MASK_STR_SHORT) and esm/utils/encoding.py (tokenize_sequence).
# MIT, Copyright 2026 Chan Zuckerberg Biohub, Inc. -- see LICENSE-esm.md. All three are unchanged
# from 3.1.6 in every part reproduced here; 3.4.0's edits to those files are elsewhere in them
# (added constants, and the biohub/ HuggingFace repo ids used by its downloader, not by us).
#
# Upstream subclasses transformers' PreTrainedTokenizerFast over a `tokenizers` BPE with no
# merges -- which, as its own comment says, is just a character-level lookup. Rewritten here as a
# plain dict lookup so the demo needs neither transformers nor tokenizers. Checked against
# upstream on 6,000 fuzz cases (random strings over the vocab, unknown characters, embedded
# "<mask>", empty string) x both add_special_tokens values: zero mismatches.

import torch

# fmt: off
SEQUENCE_VOCAB = [
    "<cls>", "<pad>", "<eos>", "<unk>",
    "L", "A", "G", "V", "S", "E", "R", "T", "I", "D", "P", "K",
    "Q", "N", "F", "Y", "M", "H", "W", "C", "X", "B", "U", "Z",
    "O", ".", "-", "|",
    "<mask>",
]
# fmt: on

MASK_STR_SHORT = "_"

# Matched greedily before falling back to single characters, the way upstream's added-token trie
# does. "|" needs no entry here: it is one character.
_SPECIAL_STRS = ("<cls>", "<pad>", "<eos>", "<unk>", "<mask>")


class EsmSequenceTokenizer:
    """Character-level ESM sequence tokenizer: `<cls> $A <eos>`, ids are vocab indices."""

    model_input_names = ["sequence_tokens", "attention_mask"]

    def __init__(
        self,
        unk_token="<unk>",
        cls_token="<cls>",
        pad_token="<pad>",
        mask_token="<mask>",
        eos_token="<eos>",
        chain_break_token="|",
    ):
        self._vocab = list(SEQUENCE_VOCAB)
        self._token_to_id = {tok: ind for ind, tok in enumerate(self._vocab)}

        self.unk_token = unk_token
        self.cls_token = cls_token
        self.pad_token = pad_token
        self.mask_token = mask_token
        self.eos_token = eos_token
        self.chain_break_token = chain_break_token
        # Upstream overrides bos to cls: "we never use the bos token anywhere".
        self.bos_token = cls_token

        self.unk_token_id = self._token_to_id[unk_token]
        self.cls_token_id = self._token_to_id[cls_token]
        self.pad_token_id = self._token_to_id[pad_token]
        self.mask_token_id = self._token_to_id[mask_token]
        self.eos_token_id = self._token_to_id[eos_token]
        self.chain_break_token_id = self._token_to_id[chain_break_token]
        self.bos_token_id = self.cls_token_id

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    @property
    def all_token_ids(self) -> list[int]:
        return list(range(self.vocab_size))

    @property
    def special_token_ids(self) -> list[int]:
        return [
            self.cls_token_id,
            self.pad_token_id,
            self.eos_token_id,
            self.unk_token_id,
            self.mask_token_id,
            self.chain_break_token_id,
        ]

    def get_vocab(self) -> dict[str, int]:
        return dict(self._token_to_id)

    def _tokenize(self, text: str) -> list[str]:
        tokens, i = [], 0
        while i < len(text):
            for special in _SPECIAL_STRS:
                if text.startswith(special, i):
                    tokens.append(special)
                    i += len(special)
                    break
            else:
                tokens.append(text[i])
                i += 1
        return tokens

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        # Unknown characters map to <unk> one for one; upstream's BPE has fuse_unk=False.
        ids = [
            self._token_to_id.get(tok, self.unk_token_id) for tok in self._tokenize(text)
        ]
        if add_special_tokens:
            ids = [self.cls_token_id] + ids + [self.eos_token_id]
        return ids

    def decode(self, token_ids, skip_special_tokens: bool = False) -> str:
        # Unused by PREpiBind. Upstream joins tokens with spaces and its one caller strips them,
        # so this returns the stripped form directly.
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        skip = set(self.special_token_ids) if skip_special_tokens else set()
        return "".join(self._vocab[int(i)] for i in token_ids if int(i) not in skip)


def get_esmc_model_tokenizers() -> EsmSequenceTokenizer:
    return EsmSequenceTokenizer()


def tokenize_sequence(
    sequence: str,
    sequence_tokenizer: EsmSequenceTokenizer,
    add_special_tokens: bool = True,
) -> torch.Tensor:
    sequence = sequence.replace(MASK_STR_SHORT, sequence_tokenizer.mask_token)
    sequence_tokens = sequence_tokenizer.encode(
        sequence, add_special_tokens=add_special_tokens
    )
    sequence_tokens = torch.tensor(sequence_tokens, dtype=torch.int64)
    return sequence_tokens
