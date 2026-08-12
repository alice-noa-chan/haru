from __future__ import annotations

import os
from pathlib import Path

# ============================================================================
# Project paths
# ============================================================================
# Scripts use these values as their defaults. Keep each experiment in its own
# RUN_NAME so checkpoints and logs from different configurations never mix.
# Keep code and large mutable artifacts separate when CFRD runs in a cloud
# container. Locally, both default to the project directory.
CODE_DIR = Path(__file__).resolve().parent
ROOT_DIR = Path(os.environ.get("HARU_STORAGE_DIR", os.environ.get("CFRD_STORAGE_DIR", CODE_DIR))).resolve()
DATA_DIR = ROOT_DIR / "data"
TOKENIZER_DIR = ROOT_DIR / "tokenizer"
RUN_NAME = "haru-v2-binding"
PACKED_DIR = ROOT_DIR / "packed" / RUN_NAME
RUN_DIR = ROOT_DIR / "runs" / RUN_NAME

TOKENIZER_MODEL_PATH = TOKENIZER_DIR / "tiny_ko_12k.model"
TOKENIZER_VOCAB_PATH = TOKENIZER_DIR / "tiny_ko_12k.vocab"
TRAIN_BIN_PATH = PACKED_DIR / "train.bin"
VAL_BIN_PATH = PACKED_DIR / "val.bin"
PACKED_META_PATH = PACKED_DIR / "meta.json"
BEST_CHECKPOINT_PATH = RUN_DIR / "best.pt"
LATEST_CHECKPOINT_PATH = RUN_DIR / "latest.pt"
TRAIN_LOG_PATH = RUN_DIR / "train_log.jsonl"
TRANSFORMERS_EXPORT_DIR = RUN_DIR / "transformers"

# ============================================================================
# Source data
# ============================================================================
# Files are discovered recursively under data/.
# - *.txt: one sample per line
# - *.jsonl: one object per line; JSONL_TEXT_KEY contains the sample
JSONL_TEXT_KEY = "text"
VAL_FRACTION = 0.005
TEXT_UNICODE_NORMALIZATION = "NFC"
SKIP_EMPTY_TEXT = True

# Preserve newlines explicitly instead of letting SentencePiece turn them into spaces.
NEWLINE_TOKEN = "<|nl|>"
# Escape a literal occurrence of NEWLINE_TOKEN so encode/decode stays lossless.
NEWLINE_ESCAPE_TOKEN = "<|literal_nl|>"

# ============================================================================
# Tokenizer
# ============================================================================
TOKENIZER_VOCAB_SIZE = 12_000
TOKENIZER_MODEL_TYPE = "unigram"
TOKENIZER_CHARACTER_COVERAGE = 1.0
TOKENIZER_BYTE_FALLBACK = True
TOKENIZER_MAX_SENTENCES = 2_000_000
TOKENIZER_MAX_SENTENCE_LENGTH = 16_384
TOKENIZER_NUM_THREADS = 16

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3

# ============================================================================
# CFRD (Causal Folded Recurrent Decoder)
# ============================================================================
# The next-generation training default stays within the 8-12 million parameter
# budget while giving tokens more independent processing capacity.
CONTEXT_LENGTH = 512
CHUNK_SIZE = 64
D_MODEL = 384
N_HEAD = 6
N_KV_HEAD = 2
FFN_DIM = 1024
ROPE_THETA = 10_000.0
DROPOUT = 0.0

# Summary memory compresses each completed chunk into a few small slots.
SUMMARY_SLOTS = 4
MEMORY_DIM = 128
MEMORY_HEADS = 4

# Initial per-head penalty for attending to increasingly old summary chunks.
# This gives long-range memory an explicit sense of event order.
MEMORY_RECENCY_BIAS_INIT = 0.10

# Three physical cells are reused in alternating order for six recurrent steps.
PHYSICAL_CELLS = 3
RECURRENCES = 6

# Finish the recurrent stack with one non-recurrent, full-context causal block.
# This gives entity-role bindings an independent path that is not forced through
# the shared local cells or compressed summary memory.
USE_BINDING_BLOCK = True

# Apply the shared LM head at recurrent depths 2, 4, and 6 during training.
# Depth 6 is the main objective; earlier depths receive auxiliary supervision.
EXIT_DEPTHS = (2, 4, 6)
AUX_EXIT_LOSS_WEIGHT = 0.15

# Keep repeatedly applied residual updates small. The model also multiplies each
# update by 1 / sqrt(RECURRENCES).
RESIDUAL_GATE_INIT = -1.0
MEMORY_GAIN_INIT = 0.0

# Add a small projection of Korean onset/vowel/coda and surface features.
USE_KOREAN_SURFACE_FEATURES = True
SURFACE_FEATURE_GAIN_INIT = 0.10

# ============================================================================
# Training
# ============================================================================
SEED = 1337
DEVICE = "auto"  # "auto", "cuda", "cpu", or "mps"
PRECISION = "bf16"  # "bf16", "fp16", "fp32"

BATCH_SIZE = 32
GRAD_ACCUM_STEPS = 8
TARGET_TOKENS = 800_000_000

LEARNING_RATE = 4.0e-4
MIN_LEARNING_RATE = 4.0e-5
WARMUP_STEPS = 200
WEIGHT_DECAY = 0.10
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.95
ADAM_EPS = 1.0e-8
GRAD_CLIP_NORM = 1.0

# Pair ordinary next-token learning with a small relation-binding objective.
# Each micro-batch covers all five relation families, while fresh entity
# permutations prevent the model from memorizing fixed name/answer identities.
COUNTERFACTUAL_LOSS_WEIGHT = 0.20
COUNTERFACTUAL_PAIRS_PER_MICRO_BATCH = 5
COUNTERFACTUAL_MARGIN = 0.50
COUNTERFACTUAL_EVAL_PAIRS = 100

LOG_INTERVAL = 10
EVAL_INTERVAL = 250
EVAL_BATCHES = 100
FINAL_EVAL_BATCHES = 200
SAVE_INTERVAL = 500

# torch.compile behavior varies by platform, so it is opt-in.
USE_TORCH_COMPILE = False

# Verify that packed data still matches the tokenizer and source files.
STRICT_DATA_FINGERPRINT = True

# ============================================================================
# Generation
# ============================================================================
GENERATION_PROMPT = "작은 마을에 조용한 아침이 찾아왔어요."
GENERATION_MAX_NEW_TOKENS = 180
GENERATION_TEMPERATURE = 0.70
GENERATION_TOP_P = 0.90
GENERATION_TOP_K = 40
GENERATION_REPETITION_PENALTY = 1.08
GENERATION_SEED = 42

# Choose 2, 4, or 6. Each depth is supervised during training.
INFERENCE_RECURSIONS = 6
