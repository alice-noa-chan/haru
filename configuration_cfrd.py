"""Hugging Face configuration for CFRD."""

from __future__ import annotations

from transformers import PretrainedConfig

try:
    from .model import ModelConfig
except ImportError:  # Direct imports from the project root.
    from model import ModelConfig


class CFRDConfig(PretrainedConfig):
    """Serializable Transformers configuration for CFRD causal language models."""

    model_type = "cfrd"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 8192,
        context_length: int = 512,
        chunk_size: int = 64,
        hidden_size: int = 384,
        num_attention_heads: int = 6,
        num_key_value_heads: int = 2,
        intermediate_size: int = 1024,
        rope_theta: float = 10_000.0,
        dropout: float = 0.0,
        summary_slots: int = 4,
        memory_dim: int = 128,
        memory_heads: int = 4,
        memory_recency_bias_init: float = 0.10,
        physical_cells: int = 2,
        recurrences: int = 6,
        exit_depths: list[int] | tuple[int, ...] = (2, 4, 6),
        aux_exit_loss_weight: float = 0.15,
        residual_gate_init: float = -1.0,
        memory_gain_init: float = 0.0,
        use_binding_block: bool = False,
        use_surface_features: bool = True,
        surface_feature_dim: int = 76,
        surface_feature_gain_init: float = 0.10,
        inference_recurrences: int | None = None,
        bos_token_id: int = 2,
        eos_token_id: int = 3,
        pad_token_id: int = 0,
        **kwargs,
    ) -> None:
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.max_position_embeddings = context_length
        self.chunk_size = chunk_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.rope_theta = rope_theta
        self.dropout = dropout
        self.summary_slots = summary_slots
        self.memory_dim = memory_dim
        self.memory_heads = memory_heads
        self.memory_recency_bias_init = memory_recency_bias_init
        self.physical_cells = physical_cells
        self.recurrences = recurrences
        self.exit_depths = list(exit_depths)
        self.aux_exit_loss_weight = aux_exit_loss_weight
        self.residual_gate_init = residual_gate_init
        self.memory_gain_init = memory_gain_init
        self.use_binding_block = use_binding_block
        self.use_surface_features = use_surface_features
        self.surface_feature_dim = surface_feature_dim
        self.surface_feature_gain_init = surface_feature_gain_init
        self.inference_recurrences = recurrences if inference_recurrences is None else inference_recurrences

        kwargs.setdefault("is_decoder", True)
        kwargs.setdefault("tie_word_embeddings", False)
        kwargs.setdefault("use_cache", False)
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            **kwargs,
        )

        self.to_model_config().validate()
        if self.inference_recurrences not in self.exit_depths:
            raise ValueError("inference_recurrences must be one of exit_depths")

    @classmethod
    def from_model_config(
        cls,
        model_config: ModelConfig,
        *,
        inference_recurrences: int | None = None,
        bos_token_id: int = 2,
        eos_token_id: int = 3,
        pad_token_id: int = 0,
    ) -> "CFRDConfig":
        """Convert the training dataclass to a Transformers configuration."""

        return cls(
            vocab_size=model_config.vocab_size,
            context_length=model_config.context_length,
            chunk_size=model_config.chunk_size,
            hidden_size=model_config.d_model,
            num_attention_heads=model_config.n_head,
            num_key_value_heads=model_config.n_kv_head,
            intermediate_size=model_config.ffn_dim,
            rope_theta=model_config.rope_theta,
            dropout=model_config.dropout,
            summary_slots=model_config.summary_slots,
            memory_dim=model_config.memory_dim,
            memory_heads=model_config.memory_heads,
            memory_recency_bias_init=model_config.memory_recency_bias_init,
            physical_cells=model_config.physical_cells,
            recurrences=model_config.recurrences,
            exit_depths=model_config.exit_depths,
            aux_exit_loss_weight=model_config.aux_exit_loss_weight,
            residual_gate_init=model_config.residual_gate_init,
            memory_gain_init=model_config.memory_gain_init,
            use_binding_block=model_config.use_binding_block,
            use_surface_features=model_config.use_surface_features,
            surface_feature_dim=model_config.surface_feature_dim,
            surface_feature_gain_init=model_config.surface_feature_gain_init,
            inference_recurrences=(
                model_config.recurrences if inference_recurrences is None else inference_recurrences
            ),
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )

    def to_model_config(self) -> ModelConfig:
        """Convert this Transformers configuration to the core model dataclass."""

        return ModelConfig(
            vocab_size=self.vocab_size,
            context_length=self.context_length,
            chunk_size=self.chunk_size,
            d_model=self.hidden_size,
            n_head=self.num_attention_heads,
            n_kv_head=self.num_key_value_heads,
            ffn_dim=self.intermediate_size,
            rope_theta=self.rope_theta,
            dropout=self.dropout,
            summary_slots=self.summary_slots,
            memory_dim=self.memory_dim,
            memory_heads=self.memory_heads,
            memory_recency_bias_init=self.memory_recency_bias_init,
            physical_cells=self.physical_cells,
            recurrences=self.recurrences,
            exit_depths=tuple(self.exit_depths),
            aux_exit_loss_weight=self.aux_exit_loss_weight,
            residual_gate_init=self.residual_gate_init,
            memory_gain_init=self.memory_gain_init,
            use_binding_block=self.use_binding_block,
            use_surface_features=self.use_surface_features,
            surface_feature_dim=self.surface_feature_dim,
            surface_feature_gain_init=self.surface_feature_gain_init,
        )


CFRDConfig.register_for_auto_class()
