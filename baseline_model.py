"""Parameter-matched dense Transformer baseline for CFRD ablations.

CFRD trades parameters for compute: three physical cells are applied six times,
so an 11.6M-parameter CFRD forward costs about 1.47x the FLOPs of any dense
decoder holding the same parameter count. A single "same parameter count"
comparison therefore favors CFRD. Two baselines are required to bracket it:

- parameter-matched  (fewer FLOPs than CFRD): does folding buy quality at all?
- compute-matched    (more parameters than CFRD): does folding buy quality per
  parameter, which is the only claim a compact model can honestly make?

This module deliberately implements an ordinary pre-norm decoder: full causal
attention, no summary memory, no recurrence, no residual gates. Only the
embedding path is shared with CFRD, so Korean surface features never become the
explanation for a difference between the two architectures.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .cfrd_features import SURFACE_FEATURE_DIM
    from .model import FullCausalAttention, ModelOutput, RMSNorm, SwiGLU, build_rope_cache
except ImportError:  # Direct script imports from the project root.
    from cfrd_features import SURFACE_FEATURE_DIM
    from model import FullCausalAttention, ModelOutput, RMSNorm, SwiGLU, build_rope_cache


@dataclass(frozen=True, slots=True)
class BaselineConfig:
    """Serializable structural configuration for the dense baseline."""

    vocab_size: int
    context_length: int = 512
    d_model: int = 384
    n_head: int = 6
    n_kv_head: int = 2
    ffn_dim: int = 1152
    n_layer: int = 4
    rope_theta: float = 10_000.0
    dropout: float = 0.0
    # Intermediate layers that also project through the language-model head.
    # Empty means one exit, an ordinary decoder. Set this to give the baseline
    # the same deep supervision CFRD gets, so a CFRD win cannot be explained by
    # auxiliary gradient signal the control never received.
    auxiliary_exit_layers: tuple[int, ...] = ()
    aux_exit_loss_weight: float = 0.15
    use_surface_features: bool = True
    surface_feature_dim: int = SURFACE_FEATURE_DIM
    surface_feature_gain_init: float = 0.10

    @property
    def exit_depths(self) -> tuple[int, ...]:
        """Supervised depths, final last, matching CFRD's interface."""

        return tuple(sorted({*self.auxiliary_exit_layers, self.n_layer}))

    def validate(self) -> None:
        positive_integers = {
            "vocab_size": self.vocab_size,
            "context_length": self.context_length,
            "d_model": self.d_model,
            "n_head": self.n_head,
            "n_kv_head": self.n_kv_head,
            "ffn_dim": self.ffn_dim,
            "n_layer": self.n_layer,
        }
        invalid = [name for name, value in positive_integers.items() if value <= 0]
        if invalid:
            raise ValueError(f"These values must be positive: {', '.join(invalid)}")
        if self.d_model % self.n_head != 0:
            raise ValueError("d_model must be divisible by n_head")
        if self.n_head % self.n_kv_head != 0:
            raise ValueError("n_head must be divisible by n_kv_head")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1)")
        if any(depth <= 0 or depth > self.n_layer for depth in self.auxiliary_exit_layers):
            raise ValueError("Every auxiliary exit layer must be between 1 and n_layer")
        if self.n_layer in self.auxiliary_exit_layers:
            raise ValueError("The final layer is always supervised and must not be listed as auxiliary")
        if len(set(self.auxiliary_exit_layers)) != len(self.auxiliary_exit_layers):
            raise ValueError("auxiliary_exit_layers must be unique")

    @classmethod
    def from_checkpoint(cls, checkpoint: dict, vocab_size: int) -> "BaselineConfig":
        """Build a baseline config from a checkpoint without trusting unrelated keys."""

        raw = checkpoint.get("model_config")
        if not isinstance(raw, dict):
            raise ValueError("Checkpoint does not contain model_config")

        allowed = {field.name for field in fields(cls)}
        values = {key: value for key, value in raw.items() if key in allowed}
        values["vocab_size"] = vocab_size
        if "auxiliary_exit_layers" in values:
            values["auxiliary_exit_layers"] = tuple(values["auxiliary_exit_layers"])
        return cls(**values)

    @classmethod
    def from_project_settings(cls, settings: object, vocab_size: int) -> "BaselineConfig":
        """Read the training project's uppercase settings without importing it here."""

        return cls(
            vocab_size=vocab_size,
            context_length=settings.CONTEXT_LENGTH,
            d_model=settings.D_MODEL,
            n_head=settings.N_HEAD,
            n_kv_head=settings.N_KV_HEAD,
            ffn_dim=settings.BASELINE_FFN_DIM,
            n_layer=settings.BASELINE_LAYERS,
            rope_theta=settings.ROPE_THETA,
            dropout=settings.DROPOUT,
            use_surface_features=settings.USE_KOREAN_SURFACE_FEATURES,
            surface_feature_dim=SURFACE_FEATURE_DIM,
            surface_feature_gain_init=settings.SURFACE_FEATURE_GAIN_INIT,
        )


class BaselineBlock(nn.Module):
    """Ordinary pre-norm decoder layer: full causal attention, then SwiGLU.

    Unlike FoldedCell there is no depth conditioning, no summary memory, and no
    residual gate. Every difference from CFRD must stay attributable.
    """

    def __init__(self, cfg: BaselineConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.attention = FullCausalAttention(cfg)
        self.ffn = SwiGLU(cfg)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attention(self.attn_norm(x), rope_cos, rope_sin)
        return x + self.ffn(self.ffn_norm(x))


class BaselineLanguageModel(nn.Module):
    """Dense causal decoder that mirrors CFRDLanguageModel's public interface."""

    def __init__(
        self,
        cfg: BaselineConfig,
        surface_feature_table: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        cfg.validate()
        self.cfg = cfg

        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.embedding_dropout = nn.Dropout(cfg.dropout)

        if cfg.use_surface_features:
            if surface_feature_table is None:
                raise ValueError("surface_feature_table is required when surface features are enabled")
            if surface_feature_table.shape != (cfg.vocab_size, cfg.surface_feature_dim):
                raise ValueError(
                    "Invalid surface_feature_table shape: "
                    f"{tuple(surface_feature_table.shape)} != "
                    f"({cfg.vocab_size}, {cfg.surface_feature_dim})"
                )

            self.register_buffer("surface_feature_table", surface_feature_table.float(), persistent=True)
            self.surface_projection = nn.Linear(cfg.surface_feature_dim, cfg.d_model, bias=False)
            self.surface_gain = nn.Parameter(torch.tensor(cfg.surface_feature_gain_init, dtype=torch.float32))
        else:
            self.register_buffer("surface_feature_table", torch.empty(0), persistent=False)
            self.surface_projection = None
            self.surface_gain = None

        self.blocks = nn.ModuleList([BaselineBlock(cfg) for _ in range(cfg.n_layer)])
        self.final_norm = RMSNorm(cfg.d_model)

        head_dim = cfg.d_model // cfg.n_head
        rope_cos, rope_sin = build_rope_cache(cfg.context_length, head_dim, cfg.rope_theta)
        self.register_buffer("rope_cos", rope_cos, persistent=True)
        self.register_buffer("rope_sin", rope_sin, persistent=True)

        self.apply(self._init_weights)
        self._init_residual_outputs()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _init_residual_outputs(self) -> None:
        # Scale residual output projections by depth, as CFRD does for its cells.
        std = 0.02 / math.sqrt(2.0 * self.cfg.n_layer)
        for block in self.blocks:
            nn.init.normal_(block.attention.o_proj.weight, mean=0.0, std=std)
            nn.init.normal_(block.ffn.w2.weight, mean=0.0, std=std)

    def _embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(token_ids)

        if self.cfg.use_surface_features:
            assert self.surface_projection is not None
            assert self.surface_gain is not None

            features = self.surface_feature_table[token_ids]
            surface = self.surface_projection(features.to(dtype=x.dtype))
            x = x + self.surface_gain.to(dtype=x.dtype) * surface

        return self.embedding_dropout(x)

    def _logits(self, x: torch.Tensor, logits_to_keep: int = 0) -> torch.Tensor:
        # Tie the LM head to token embeddings to avoid a second vocabulary matrix.
        if logits_to_keep < 0:
            raise ValueError("logits_to_keep must be non-negative")
        if logits_to_keep:
            x = x[:, -logits_to_keep:, :]
        return F.linear(self.final_norm(x), self.token_embedding.weight)

    def forward(
        self,
        token_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        recurrences: int | None = None,
        logits_to_keep: int = 0,
    ) -> ModelOutput:
        batch, time = token_ids.shape
        del batch

        if time > self.cfg.context_length:
            raise ValueError(f"Sequence length {time} exceeds context_length {self.cfg.context_length}")
        # Accepted only so shared training and evaluation code can stay uniform.
        # A dense stack has no reusable depth to vary, so silently ignoring a
        # different value would fake a recurrent-depth comparison.
        if recurrences is not None and recurrences != self.cfg.n_layer:
            raise ValueError(
                f"The dense baseline runs exactly {self.cfg.n_layer} layers and has no shallower supervised exit"
            )
        if targets is not None and logits_to_keep:
            raise ValueError("logits_to_keep cannot be used when targets are provided")

        x = self._embed(token_ids)

        exit_losses: dict[int, torch.Tensor] = {}
        auxiliary_depths = set(self.cfg.auxiliary_exit_layers)
        final_logits: torch.Tensor | None = None

        for index, block in enumerate(self.blocks):
            x = block(x, self.rope_cos, self.rope_sin)
            depth = index + 1

            is_final = depth == self.cfg.n_layer
            # Auxiliary heads exist only to shape training. Skipping them at
            # inference keeps the eval-time cost of this arm honest.
            if not is_final and not (targets is not None and depth in auxiliary_depths):
                continue

            depth_logits = self._logits(x, logits_to_keep if is_final else 0)
            if is_final:
                final_logits = depth_logits
            if targets is not None:
                exit_losses[depth] = F.cross_entropy(
                    depth_logits.reshape(-1, depth_logits.size(-1)),
                    targets.reshape(-1),
                )

        assert final_logits is not None

        final_loss: torch.Tensor | None = None
        total_loss: torch.Tensor | None = None

        if targets is not None:
            final_loss = exit_losses[self.cfg.n_layer]
            auxiliary = [value for depth, value in exit_losses.items() if depth != self.cfg.n_layer]
            if auxiliary:
                total_loss = final_loss + self.cfg.aux_exit_loss_weight * torch.stack(auxiliary).mean()
            else:
                total_loss = final_loss

        return ModelOutput(logits=final_logits, loss=total_loss, final_loss=final_loss, exit_losses=exit_losses)
