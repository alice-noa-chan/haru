"""Portable AutoModelForCausalLM wrapper with a legacy tuple KV cache."""

import torch
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

try:
    from .configuration_haru_dense import HaruDenseConfig
    from .dense_model import DenseConfig, DenseLanguageModel
except ImportError:
    from configuration_haru_dense import HaruDenseConfig
    from dense_model import DenseConfig, DenseLanguageModel


class HaruDenseForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = HaruDenseConfig
    base_model_prefix = "model"
    main_input_name = "input_ids"
    _supports_cache_class = False

    def __init__(self, config):
        super().__init__(config)
        cfg = DenseConfig(**config.dense_config)
        table = torch.zeros(cfg.vocab_size, cfg.surface_feature_dim) if cfg.use_surface_features else None
        self.model = DenseLanguageModel(cfg, table)

    @classmethod
    def _supports_default_dynamic_cache(cls):
        return False

    def get_input_embeddings(self):
        return self.model.token_embedding

    def get_output_embeddings(self):
        return None

    def set_input_embeddings(self, value):
        self.model.token_embedding = value

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        labels=None,
        past_key_values=None,
        use_cache=None,
        logits_to_keep=0,
        return_dict=None,
        **kwargs,
    ):
        if labels is not None and logits_to_keep:
            raise ValueError("Full logits required for labels")
        output = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=self.config.use_cache if use_cache is None else use_cache,
            logits_to_keep=logits_to_keep,
        )
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                output.logits[:, :-1].reshape(-1, self.config.vocab_size), labels[:, 1:].reshape(-1), ignore_index=-100
            )
        if return_dict is False:
            values = (output.logits, output.past_key_values)
            return (loss, *values) if loss is not None else values
        return CausalLMOutputWithPast(loss=loss, logits=output.logits, past_key_values=output.past_key_values)

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, past_key_values=None, **kwargs):
        if past_key_values is not None:
            cached = past_key_values[0][0].shape[2]
            input_ids = input_ids[:, cached:] if input_ids.shape[1] > cached else input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
            "logits_to_keep": 1,
        }

    def generate(self, inputs=None, generation_config=None, **kwargs):
        ids = inputs if inputs is not None else kwargs.get("input_ids")
        config = generation_config or self.generation_config
        extra = kwargs.get("max_new_tokens", config.max_new_tokens)
        requested = (
            ids.shape[1] + extra
            if ids is not None and extra is not None
            else kwargs.get("max_length", config.max_length)
        )
        if requested > self.model.cfg.context_length:
            raise ValueError("Input plus requested generation exceeds 1024/context_length; shorten it explicitly")
        return super().generate(inputs=inputs, generation_config=generation_config, **kwargs)

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        return tuple(
            tuple(value.index_select(0, beam_idx.to(value.device)) for value in layer) for layer in past_key_values
        )


HaruDenseForCausalLM.register_for_auto_class("AutoModelForCausalLM")
