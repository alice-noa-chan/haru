"""Hugging Face tokenizer for CFRD SentencePiece models."""

from __future__ import annotations

import os
import shutil
import unicodedata
from pathlib import Path

import sentencepiece as spm
from transformers import PreTrainedTokenizer

VOCAB_FILES_NAMES = {"vocab_file": "tokenizer.model"}


class CFRDTokenizer(PreTrainedTokenizer):
    """SentencePiece tokenizer with lossless newline preservation."""

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file: str,
        newline_token: str = "<|nl|>",
        newline_escape_token: str = "<|literal_nl|>",
        add_bos_token: bool = True,
        add_eos_token: bool = False,
        **kwargs,
    ) -> None:
        self.vocab_file = vocab_file
        self.newline_token = newline_token
        self.newline_escape_token = newline_escape_token
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self.sp_model = spm.SentencePieceProcessor(model_file=str(vocab_file))
        super().__init__(**kwargs)

    @property
    def vocab_size(self) -> int:
        return int(self.sp_model.vocab_size())

    def get_vocab(self) -> dict[str, int]:
        vocabulary = {self.convert_ids_to_tokens(index): index for index in range(self.vocab_size)}
        vocabulary.update(self.get_added_vocab())
        return vocabulary

    def _tokenize(self, text: str) -> list[str]:
        text = unicodedata.normalize("NFC", text)
        text = text.replace(self.newline_token, self.newline_escape_token)
        text = text.replace("\n", self.newline_token)
        return list(self.sp_model.encode(text, out_type=str))

    def _convert_token_to_id(self, token: str) -> int:
        return int(self.sp_model.piece_to_id(token))

    def _convert_id_to_token(self, index: int) -> str:
        return str(self.sp_model.id_to_piece(index))

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        text = self.sp_model.decode(tokens)
        text = text.replace(self.newline_token, "\n")
        return text.replace(self.newline_escape_token, self.newline_token)

    def build_inputs_with_special_tokens(
        self,
        token_ids_0: list[int],
        token_ids_1: list[int] | None = None,
    ) -> list[int]:
        if token_ids_1 is not None:
            raise ValueError("CFRDTokenizer does not support paired sequences")
        output = list(token_ids_0)
        if self.add_bos_token:
            output.insert(0, self.bos_token_id)
        if self.add_eos_token:
            output.append(self.eos_token_id)
        return output

    def get_special_tokens_mask(
        self,
        token_ids_0: list[int],
        token_ids_1: list[int] | None = None,
        already_has_special_tokens: bool = False,
    ) -> list[int]:
        if already_has_special_tokens:
            special_ids = set(self.all_special_ids)
            return [1 if token_id in special_ids else 0 for token_id in token_ids_0]
        if token_ids_1 is not None:
            raise ValueError("CFRDTokenizer does not support paired sequences")
        return ([1] if self.add_bos_token else []) + [0] * len(token_ids_0) + ([1] if self.add_eos_token else [])

    def create_token_type_ids_from_sequences(
        self,
        token_ids_0: list[int],
        token_ids_1: list[int] | None = None,
    ) -> list[int]:
        return [0] * len(self.build_inputs_with_special_tokens(token_ids_0, token_ids_1))

    def save_vocabulary(self, save_directory: str, filename_prefix: str | None = None):
        directory = Path(save_directory)
        directory.mkdir(parents=True, exist_ok=True)
        filename = ((filename_prefix + "-") if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        destination = directory / filename
        if os.path.abspath(self.vocab_file) != os.path.abspath(destination):
            shutil.copyfile(self.vocab_file, destination)
        return (str(destination),)


CFRDTokenizer.register_for_auto_class("AutoTokenizer")
