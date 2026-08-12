from __future__ import annotations

import math
import unicodedata

import torch

from cfrd_features import N_CHO, N_JONG, N_JUNG, SURFACE_FEATURE_DIM
from tokenizer_utils import StoryTokenizer


def _is_punctuation(char: str) -> bool:
    return unicodedata.category(char).startswith("P")


def _is_hangul_syllable(char: str) -> bool:
    code = ord(char)
    return 0xAC00 <= code <= 0xD7A3


def _decompose_hangul(char: str) -> tuple[int, int, int]:
    """Split one precomposed Hangul syllable into onset, vowel, and coda IDs."""

    offset = ord(char) - 0xAC00
    cho = offset // (N_JUNG * N_JONG)
    jung = (offset % (N_JUNG * N_JONG)) // N_JONG
    jong = offset % N_JONG
    return cho, jung, jong


def build_surface_feature_table(tokenizer: StoryTokenizer) -> torch.Tensor:
    """
    Convert every vocabulary piece into compact orthographic features.

    These features use only the token string, never corpus statistics. Similar
    Korean subwords can therefore share onset/vowel/coda structure instead of
    starting from completely unrelated embeddings.
    """

    table = torch.zeros(tokenizer.vocab_size, SURFACE_FEATURE_DIM, dtype=torch.float32)

    extra_base = N_CHO + N_JUNG + N_JONG

    for token_id in range(tokenizer.vocab_size):
        piece = tokenizer.id_to_piece(token_id)
        feature = table[token_id]

        # SentencePiece uses this marker for a word boundary.
        starts_with_boundary = piece.startswith("▁")
        clean_piece = piece.replace("▁", "")

        # Byte-fallback pieces use the form <0xAB>.
        is_byte_piece = clean_piece.startswith("<0x") and clean_piece.endswith(">")
        is_special_piece = clean_piece.startswith("<") and clean_piece.endswith(">") and not is_byte_piece

        hangul_count = 0
        has_digit = False
        has_ascii = False
        has_punctuation = False
        has_other = False

        if not is_special_piece and not is_byte_piece:
            for char in clean_piece:
                if _is_hangul_syllable(char):
                    cho, jung, jong = _decompose_hangul(char)
                    feature[cho] += 1.0
                    feature[N_CHO + jung] += 1.0
                    feature[N_CHO + N_JUNG + jong] += 1.0
                    hangul_count += 1
                elif char.isdigit():
                    has_digit = True
                elif ord(char) < 128 and char.isprintable():
                    has_ascii = True
                    if _is_punctuation(char):
                        has_punctuation = True
                elif _is_punctuation(char):
                    has_punctuation = True
                elif not char.isspace():
                    has_other = True

        # Average the Jamo histogram so magnitude does not grow with token length.
        if hangul_count > 0:
            feature[: N_CHO + N_JUNG + N_JONG] /= float(hangul_count)

        # Additional features.
        feature[extra_base + 0] = 1.0 if starts_with_boundary else 0.0
        feature[extra_base + 1] = 1.0 if hangul_count > 0 else 0.0
        feature[extra_base + 2] = 1.0 if has_digit else 0.0
        feature[extra_base + 3] = 1.0 if has_ascii else 0.0
        feature[extra_base + 4] = 1.0 if has_punctuation else 0.0
        feature[extra_base + 5] = 1.0 if has_other else 0.0
        feature[extra_base + 6] = 1.0 if is_byte_piece else 0.0

        # Log-scaled length prevents long pieces from dominating the projection.
        piece_length = max(0, len(clean_piece))
        feature[extra_base + 7] = math.log1p(min(piece_length, 32)) / math.log1p(32)

    return table
