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
RUN_NAME = "haru-v2-unfolded"
RELEASE_VERSION = "2.0"
HUGGINGFACE_MODEL_ID = "gaon12/haru_2.0"
PREVIOUS_HUGGINGFACE_MODEL_ID = "gaon12/haru_1.1"
PACKED_DIR = ROOT_DIR / "packed" / RUN_NAME
RUN_DIR = ROOT_DIR / "runs" / RUN_NAME

TOKENIZER_MODEL_PATH = TOKENIZER_DIR / "haru_v2_12k.model"
TOKENIZER_VOCAB_PATH = TOKENIZER_DIR / "haru_v2_12k.vocab"
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
# BPE, measured. It beat unigram at every vocabulary size tried on the actual
# corpus (2.056 vs 2.114 chars/token at 12k, 2.326 vs 2.401 at 24k, 2.413 vs
# 2.502 at 32k), so the usual "Korean is agglutinative, prefer unigram" did not
# hold here. See results/tokenizer_comparison.json.
TOKENIZER_MODEL_TYPE = "bpe"
# Below 1.0 because the mixed corpus contains 36,970 distinct characters,
# mostly Han, emoji and symbols, and full coverage would demand a vocabulary
# entry for every one of them: SentencePiece refuses outright at 12,000.
# Rare characters fall through to byte_fallback, which is what it is for.
TOKENIZER_CHARACTER_COVERAGE = 0.9995
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
# Trimmed from 1024 so the unfolded model lands at 16,983,213 parameters,
# inside the 17M ceiling that 1024 misses by 47,725.
FFN_DIM = 1016
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
# Unfolded: one cell per recurrence. Sharing cells lost by 0.0583 validation
# loss beyond the paired spread at release scale, and by a similar margin at
# compact scale. It is the only result that survived every scale tested.
PHYSICAL_CELLS = 6
RECURRENCES = 6

# Finish the recurrent stack with one non-recurrent, full-context causal block.
# This gives entity-role bindings an independent path that is not forced through
# the shared local cells or compressed summary memory.
USE_BINDING_BLOCK = True

# Apply the shared LM head at recurrent depths 2, 4, and 6 during training.
# Depth 6 is the main objective; earlier depths receive auxiliary supervision.
EXIT_DEPTHS = (2, 4, 6)
# Zero, measured. Deep supervision helped the dense control and hurt this
# model on both axes: removing it improved validation loss by 0.0054 beyond
# the spread and strict pair accuracy from 0.263 to 0.377.
AUX_EXIT_LOSS_WEIGHT = 0.0

# Keep repeatedly applied residual updates small. The model also multiplies each
# update by 1 / sqrt(RECURRENCES).
RESIDUAL_GATE_INIT = -1.0
MEMORY_GAIN_INIT = 0.0

# Add a small projection of Korean onset/vowel/coda and surface features.
USE_KOREAN_SURFACE_FEATURES = True
SURFACE_FEATURE_GAIN_INIT = 0.10

# ============================================================================
# Dense Transformer baseline
# ============================================================================
# "cfrd" trains the folded recurrent architecture above. "dense-baseline"
# trains the control in baseline_model.py using D_MODEL, N_HEAD, N_KV_HEAD,
# CONTEXT_LENGTH, and the settings below. Always pair a change here with a new
# RUN_NAME: train.py refuses to resume across architectures, but a fresh run
# would otherwise write its checkpoints beside an unrelated experiment.
MODEL_ARCH = "cfrd"

# CFRD reuses three cells six times, so it spends about 1.45x the forward FLOPs
# of a dense decoder holding the same parameter count. Quoting a single
# "same parameter count" win would therefore compare unequal compute budgets.
# Two baselines bracket CFRD instead, both reusing D_MODEL, N_HEAD, N_KV_HEAD,
# and CONTEXT_LENGTH. Measured against Haru v1.1 CFRD, which is 11,634,459
# parameters and 17.31 GFLOPs per 512-token forward:
#
#   parameter-matched   4 layers x ffn 1184 -> 11,669,377 params, 11.95 GFLOPs
#                       parameters +0.30%, FLOPs -30.98%
#   compute-matched     7 layers x ffn 1184 -> 16,943,233 params, 17.34 GFLOPs
#                       parameters +45.63%, FLOPs +0.21%
#
# ffn_dim stays at 1184 (3.08x D_MODEL, matching CFRD's own cell proportion) in
# both arms. Matching parameters, depth, and FFN ratio at once is impossible;
# the parameter-and-depth matched option needs ffn 512 (1.33x D_MODEL), which
# would handicap the baseline rather than control for the architecture.
#
# These are the values compare_architectures.py derives automatically. Set them
# here only to train a single baseline arm through train.py.
BASELINE_LAYERS = 4
BASELINE_FFN_DIM = 1184

# ============================================================================
# Training
# ============================================================================
SEED = 1337
DEVICE = "auto"  # "auto", "cuda", "cpu", or "mps"
PRECISION = "bf16"  # "bf16", "fp16", "fp32"

BATCH_SIZE = 32
GRAD_ACCUM_STEPS = 8
# Four passes over the 2,598,939,881 packed tokens, at 612 tokens per
# parameter. Chinchilla's 20:1 answers a different question, which is how to
# split a fixed compute budget between parameters and data; with the parameter
# count fixed at 17M, the model is far from saturated at 20:1. v1.1 scored at
# chance on every KoBEST task after 47 tokens per parameter.
#
# Four is the ceiling rather than the appetite: repeated data holds most of its
# value to roughly four epochs and falls away sharply after, so a fifth pass
# buys much less than the first did. Overfitting is not the binding constraint
# here, since 17M parameters cannot memorize 2.6B tokens; underfitting is.
#
# Windows are sampled with replacement (get_random_batch), so this is a token
# budget rather than four ordered passes, and repeats fall randomly through
# training instead of in blocks.
TARGET_TOKENS = 10_395_759_524

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
# Keep relation sampling independent from packed-LM window sampling so changing
# either objective does not silently change the other objective's data order.
COUNTERFACTUAL_SEED_OFFSET = 100_000
# Round the relation batch's time dimension up to this stride. Prompt lengths
# vary, so without it the model sees a new shape almost every step (16 distinct
# shapes in 40 draws) while the language-model batch stays fixed, and
# torch.compile recompiles continuously instead of paying off. Attention is
# causal and score_mask excludes the padding, so scores are unchanged.
COUNTERFACTUAL_PAD_MULTIPLE = 64

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
