"""Transformers configuration for the independent Haru v3 architecture."""

from dataclasses import asdict

from transformers import PretrainedConfig

try:
    from .dense_model import DenseConfig
except ImportError:
    from dense_model import DenseConfig


class HaruDenseConfig(PretrainedConfig):
    model_type = "haru_dense"

    def __init__(self, dense_config=None, **kwargs):
        kwargs.setdefault("bos_token_id", 2)
        kwargs.setdefault("eos_token_id", 3)
        kwargs.setdefault("pad_token_id", 0)
        kwargs.setdefault("tie_word_embeddings", True)
        super().__init__(**kwargs)
        self.dense_config = dense_config or asdict(DenseConfig())
        self.vocab_size = self.dense_config["vocab_size"]
        self.hidden_size = self.dense_config["d_model"]
        self.num_hidden_layers = self.dense_config["n_layer"]
        self.num_attention_heads = self.dense_config["n_head"]
        self.max_position_embeddings = self.dense_config["context_length"]
        self.use_cache = True


HaruDenseConfig.register_for_auto_class()
