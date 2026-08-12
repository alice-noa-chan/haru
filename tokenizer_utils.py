from __future__ import annotations

from pathlib import Path

import sentencepiece as spm

import config
from data_utils import blake2b_file, prepare_text_for_tokenizer, restore_text_from_tokenizer


class StoryTokenizer:
    """Small, consistent wrapper around the project's SentencePiece model."""

    def __init__(self, model_path: Path | None = None) -> None:
        path = model_path or config.TOKENIZER_MODEL_PATH
        if not path.exists():
            raise FileNotFoundError(f"Tokenizer model does not exist: {path}\nRun `python tokenizer_train.py` first.")

        self.model_path = path
        self.sp = spm.SentencePieceProcessor(model_file=str(path))

        self._validate_special_ids()

    def _validate_special_ids(self) -> None:
        expected = {
            "pad": config.PAD_ID,
            "unk": config.UNK_ID,
            "bos": config.BOS_ID,
            "eos": config.EOS_ID,
        }
        actual = {
            "pad": self.sp.pad_id(),
            "unk": self.sp.unk_id(),
            "bos": self.sp.bos_id(),
            "eos": self.sp.eos_id(),
        }

        if actual != expected:
            raise ValueError(f"Tokenizer special IDs differ from config.py: {actual} != {expected}")

    @property
    def vocab_size(self) -> int:
        return int(self.sp.vocab_size())

    @property
    def bos_id(self) -> int:
        return int(self.sp.bos_id())

    @property
    def eos_id(self) -> int:
        return int(self.sp.eos_id())

    @property
    def unk_id(self) -> int:
        return int(self.sp.unk_id())

    @property
    def pad_id(self) -> int:
        return int(self.sp.pad_id())

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        prepared = prepare_text_for_tokenizer(text)
        ids = list(self.sp.encode(prepared, out_type=int))

        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        text = self.sp.decode(ids)
        return restore_text_from_tokenizer(text)

    def id_to_piece(self, token_id: int) -> str:
        return str(self.sp.id_to_piece(token_id))

    def validate_checkpoint(self, checkpoint: dict) -> None:
        """Ensure a checkpoint uses this exact tokenizer, not only the same size."""

        expected_hash = checkpoint.get("tokenizer_blake2b16")
        if not isinstance(expected_hash, str):
            raise ValueError("Checkpoint does not contain a tokenizer fingerprint")
        if expected_hash != blake2b_file(self.model_path):
            raise ValueError("Checkpoint was trained with a different tokenizer")
