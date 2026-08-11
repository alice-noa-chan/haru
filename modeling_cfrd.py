"""Hugging Face AutoModelForCausalLM implementation for CFRD."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

try:
    from .configuration_cfrd import CFRDConfig
    from .model import CFRDLanguageModel
except ImportError:  # Direct imports from the project root.
    from configuration_cfrd import CFRDConfig
    from model import CFRDLanguageModel


class CFRDPreTrainedModel(PreTrainedModel):
    """Shared Transformers metadata for CFRD model classes."""

    config_class = CFRDConfig
    base_model_prefix = "model"
    main_input_name = "input_ids"
    _supports_sdpa = True

    def _init_weights(self, module) -> None:
        # CFRDLanguageModel performs architecture-specific initialization itself.
        return None


class CFRDForCausalLM(CFRDPreTrainedModel, GenerationMixin):
    """Transformers-compatible CFRD causal language model."""

    def __init__(self, config: CFRDConfig) -> None:
        super().__init__(config)
        model_config = config.to_model_config()
        surface_features = torch.zeros(
            model_config.vocab_size,
            model_config.surface_feature_dim,
            dtype=torch.float32,
        )
        self.model = CFRDLanguageModel(model_config, surface_features)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.token_embedding

    def set_input_embeddings(self, value) -> None:
        self.model.token_embedding = value

    def get_output_embeddings(self):
        # The core model applies the input embedding weight as its tied LM head.
        return None

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        recurrences: int | None = None,
        past_key_values=None,
        use_cache: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast | tuple:
        """Run CFRD with the standard causal-language-model interface."""

        del past_key_values, use_cache, kwargs
        run_recurrences = self.config.inference_recurrences if recurrences is None else recurrences

        if attention_mask is None or bool(torch.all(attention_mask == 1)):
            logits = self.model(input_ids, recurrences=run_recurrences).logits
        else:
            logits = self._forward_padded_batch(input_ids, attention_mask, run_recurrences)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return_dict = self.config.return_dict if return_dict is None else return_dict
        if not return_dict:
            return ((loss, logits) if loss is not None else (logits,))
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=None)

    def _forward_padded_batch(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        recurrences: int,
    ) -> torch.Tensor:
        """Handle variable-length padded batches without exposing padding to attention."""

        batch_size, padded_length = input_ids.shape
        rows: list[torch.Tensor] = []
        for row_index in range(batch_size):
            valid_positions = attention_mask[row_index].bool()
            row_ids = input_ids[row_index, valid_positions].unsqueeze(0)
            if row_ids.numel() == 0:
                raise ValueError("Every input row must contain at least one non-padding token")
            row_logits = self.model(row_ids, recurrences=recurrences).logits.squeeze(0)
            padded_logits = row_logits.new_zeros(padded_length, row_logits.size(-1))
            padded_logits[valid_positions] = row_logits
            rows.append(padded_logits)
        return torch.stack(rows, dim=0)

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor | bool | int | None]:
        """Keep only the active context because CFRD does not expose a cache yet."""

        recurrences = kwargs.get("recurrences")
        input_ids = input_ids[:, -self.config.context_length :]
        if attention_mask is not None:
            attention_mask = attention_mask[:, -self.config.context_length :]
        model_inputs: dict[str, torch.Tensor | bool | int | None] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": False,
        }
        if recurrences is not None:
            model_inputs["recurrences"] = recurrences
        return model_inputs


CFRDForCausalLM.register_for_auto_class("AutoModelForCausalLM")
