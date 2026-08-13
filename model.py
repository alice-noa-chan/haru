from __future__ import annotations

import math
from dataclasses import dataclass, fields

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .cfrd_features import SURFACE_FEATURE_DIM
except ImportError:  # Direct script imports from the project root.
    from cfrd_features import SURFACE_FEATURE_DIM


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Serializable structural configuration for CFRD."""

    vocab_size: int
    context_length: int = 512
    chunk_size: int = 64
    d_model: int = 384
    n_head: int = 6
    n_kv_head: int = 2
    ffn_dim: int = 1024
    rope_theta: float = 10_000.0
    dropout: float = 0.0
    summary_slots: int = 4
    memory_dim: int = 128
    memory_heads: int = 4
    memory_recency_bias_init: float = 0.10
    physical_cells: int = 2
    recurrences: int = 6
    exit_depths: tuple[int, ...] = (2, 4, 6)
    aux_exit_loss_weight: float = 0.15
    residual_gate_init: float = -1.0
    memory_gain_init: float = 0.0
    use_binding_block: bool = False
    use_surface_features: bool = True
    surface_feature_dim: int = SURFACE_FEATURE_DIM
    surface_feature_gain_init: float = 0.10

    def validate(self) -> None:
        positive_integers = {
            "vocab_size": self.vocab_size,
            "context_length": self.context_length,
            "chunk_size": self.chunk_size,
            "d_model": self.d_model,
            "n_head": self.n_head,
            "n_kv_head": self.n_kv_head,
            "ffn_dim": self.ffn_dim,
            "summary_slots": self.summary_slots,
            "memory_dim": self.memory_dim,
            "memory_heads": self.memory_heads,
            "physical_cells": self.physical_cells,
            "recurrences": self.recurrences,
        }
        invalid = [name for name, value in positive_integers.items() if value <= 0]
        if invalid:
            raise ValueError(f"These values must be positive: {', '.join(invalid)}")
        if self.context_length % self.chunk_size != 0:
            raise ValueError("context_length must be divisible by chunk_size")
        if self.d_model % self.n_head != 0:
            raise ValueError("d_model must be divisible by n_head")
        if self.n_head % self.n_kv_head != 0:
            raise ValueError("n_head must be divisible by n_kv_head")
        if self.memory_dim % self.memory_heads != 0:
            raise ValueError("memory_dim must be divisible by memory_heads")
        if not self.exit_depths:
            raise ValueError("exit_depths cannot be empty")
        if self.recurrences not in self.exit_depths:
            raise ValueError("The final recurrence must be included in exit_depths")
        if any(depth <= 0 or depth > self.recurrences for depth in self.exit_depths):
            raise ValueError("Every exit depth must be between 1 and recurrences")
        if tuple(sorted(set(self.exit_depths))) != self.exit_depths:
            raise ValueError("exit_depths must be unique and sorted")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1)")
        if self.memory_recency_bias_init <= 0.0:
            raise ValueError("memory_recency_bias_init must be positive")

    @classmethod
    def from_checkpoint(cls, checkpoint: dict, vocab_size: int) -> "ModelConfig":
        """Build a model config from a checkpoint without trusting unrelated keys."""

        raw = checkpoint.get("model_config")
        if not isinstance(raw, dict):
            raise ValueError("Checkpoint does not contain model_config")

        allowed = {field.name for field in fields(cls)}
        values = {key: value for key, value in raw.items() if key in allowed}
        values["vocab_size"] = vocab_size
        if "exit_depths" in values:
            values["exit_depths"] = tuple(values["exit_depths"])
        return cls(**values)

    @classmethod
    def from_project_settings(cls, settings: object, vocab_size: int) -> "ModelConfig":
        """Read the training project's uppercase settings without importing it here."""

        return cls(
            vocab_size=vocab_size,
            context_length=settings.CONTEXT_LENGTH,
            chunk_size=settings.CHUNK_SIZE,
            d_model=settings.D_MODEL,
            n_head=settings.N_HEAD,
            n_kv_head=settings.N_KV_HEAD,
            ffn_dim=settings.FFN_DIM,
            rope_theta=settings.ROPE_THETA,
            dropout=settings.DROPOUT,
            summary_slots=settings.SUMMARY_SLOTS,
            memory_dim=settings.MEMORY_DIM,
            memory_heads=settings.MEMORY_HEADS,
            memory_recency_bias_init=settings.MEMORY_RECENCY_BIAS_INIT,
            physical_cells=settings.PHYSICAL_CELLS,
            recurrences=settings.RECURRENCES,
            exit_depths=tuple(settings.EXIT_DEPTHS),
            aux_exit_loss_weight=settings.AUX_EXIT_LOSS_WEIGHT,
            residual_gate_init=settings.RESIDUAL_GATE_INIT,
            memory_gain_init=settings.MEMORY_GAIN_INIT,
            use_binding_block=getattr(settings, "USE_BINDING_BLOCK", False),
            use_surface_features=settings.USE_KOREAN_SURFACE_FEATURES,
            surface_feature_dim=SURFACE_FEATURE_DIM,
            surface_feature_gain_init=settings.SURFACE_FEATURE_GAIN_INIT,
        )


@dataclass(slots=True)
class ModelOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None
    final_loss: torch.Tensor | None
    exit_losses: dict[int, torch.Tensor]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute normalization in fp32 to reduce low-precision error.
        x_float = x.float()
        normalized = x_float * torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normalized.to(dtype=x.dtype) * self.weight


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    theta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if head_dim % 2 != 0:
        raise ValueError("RoPE head_dim must be even")

    freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(seq_len).float()
    angles = torch.outer(positions, freq)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    x   : [batch_like, heads, time, head_dim]
    cos : [batch_like or 1, 1, time, head_dim / 2]
    sin : [batch_like or 1, 1, time, head_dim / 2]
    """

    x_float = x.float()
    even = x_float[..., 0::2]
    odd = x_float[..., 1::2]

    output = torch.empty_like(x_float)
    output[..., 0::2] = even * cos - odd * sin
    output[..., 1::2] = even * sin + odd * cos
    return output.to(dtype=x.dtype)


def phase_rms_norm(
    x: torch.Tensor,
    norm: RMSNorm,
    scale: torch.Tensor,
    shift: torch.Tensor,
) -> torch.Tensor:
    """
    FiLM-style conditioning lets a shared cell behave differently at each depth.

    Zero-initialized scale and shift make this an ordinary RMSNorm at startup.
    """

    y = norm(x)
    return y * (1.0 + 0.1 * torch.tanh(scale)) + 0.1 * shift


class LocalCausalAttention(nn.Module):
    """Causal grouped-query attention within fixed-size chunks."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()

        self.chunk_size = cfg.chunk_size
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.head_dim = cfg.d_model // cfg.n_head
        self.dropout = cfg.dropout

        self.q_proj = nn.Linear(cfg.d_model, cfg.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> torch.Tensor:
        batch, time, channels = x.shape
        chunk = self.chunk_size
        chunk_count = math.ceil(time / chunk)
        padded_time = chunk_count * chunk
        pad_tokens = padded_time - time

        if pad_tokens:
            x = F.pad(x, (0, 0, 0, pad_tokens))

        # [B, C, W, D] -> [B*C, W, D]
        x_chunks = x.view(batch, chunk_count, chunk, channels).reshape(batch * chunk_count, chunk, channels)

        q = self.q_proj(x_chunks)
        k = self.k_proj(x_chunks)
        v = self.v_proj(x_chunks)

        q = q.view(batch * chunk_count, chunk, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch * chunk_count, chunk, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = v.view(batch * chunk_count, chunk, self.n_kv_head, self.head_dim).transpose(1, 2)

        # Keep global RoPE positions instead of resetting positions per chunk.
        cos = rope_cos[:padded_time].view(chunk_count, chunk, -1)
        sin = rope_sin[:padded_time].view(chunk_count, chunk, -1)
        cos = cos.unsqueeze(0).expand(batch, -1, -1, -1).reshape(batch * chunk_count, 1, chunk, -1)
        sin = sin.unsqueeze(0).expand(batch, -1, -1, -1).reshape(batch * chunk_count, 1, chunk, -1)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
            enable_gqa=self.n_head != self.n_kv_head,
        )

        y = y.transpose(1, 2).contiguous().view(batch * chunk_count, chunk, channels)
        y = self.o_proj(y)
        y = y.view(batch, chunk_count, chunk, channels).reshape(batch, padded_time, channels)
        return y[:, :time, :]


class FullCausalAttention(nn.Module):
    """One full-context causal attention pass used for final variable binding."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()

        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.head_dim = cfg.d_model // cfg.n_head
        self.dropout = cfg.dropout

        self.q_proj = nn.Linear(cfg.d_model, cfg.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> torch.Tensor:
        batch, time, channels = x.shape

        q = self.q_proj(x).view(batch, time, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, time, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, time, self.n_kv_head, self.head_dim).transpose(1, 2)

        cos = rope_cos[:time].view(1, 1, time, -1)
        sin = rope_sin[:time].view(1, 1, time, -1)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
            enable_gqa=self.n_head != self.n_kv_head,
        )
        y = y.transpose(1, 2).contiguous().view(batch, time, channels)
        return self.o_proj(y)


class CausalSummaryMemory(nn.Module):
    """
    Compress each chunk into summary slots and read only completed earlier chunks.

    A summary may see its complete source chunk, but it is visible only to later
    chunks. No token can use memory to see the future of its own chunk.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()

        self.chunk_size = cfg.chunk_size
        self.summary_slots = cfg.summary_slots
        self.memory_dim = cfg.memory_dim
        self.memory_heads = cfg.memory_heads
        self.memory_head_dim = cfg.memory_dim // cfg.memory_heads
        self.dropout = cfg.dropout
        self.max_chunks = cfg.context_length // cfg.chunk_size

        # Learned pooling from one chunk to a small set of summary slots.
        self.summary_queries = nn.Parameter(torch.empty(cfg.summary_slots, cfg.memory_dim))
        self.summary_k = nn.Linear(cfg.d_model, cfg.memory_dim, bias=False)
        self.summary_v = nn.Linear(cfg.d_model, cfg.memory_dim, bias=False)

        # Small multi-head attention from tokens to earlier summaries.
        self.read_q = nn.Linear(cfg.d_model, cfg.memory_dim, bias=False)
        self.read_k = nn.Linear(cfg.memory_dim, cfg.memory_dim, bias=False)
        self.read_v = nn.Linear(cfg.memory_dim, cfg.memory_dim, bias=False)
        self.read_o = nn.Linear(cfg.memory_dim, cfg.d_model, bias=False)

        # The first chunk attends to this null item so softmax always has one key.
        self.null_memory = nn.Parameter(torch.zeros(1, 1, cfg.memory_dim))

        # Per-head decay gives otherwise position-free summaries a sense of order.
        initial_decay = torch.full((cfg.memory_heads,), cfg.memory_recency_bias_init)
        self.recency_decay_raw = nn.Parameter(torch.log(torch.expm1(initial_decay)))

        source_chunks = torch.arange(self.max_chunks).repeat_interleave(cfg.summary_slots)
        source_chunks = torch.cat((torch.tensor([-1]), source_chunks))
        target_chunks = torch.arange(cfg.context_length) // cfg.chunk_size
        allowed = source_chunks.view(1, -1) < target_chunks.view(-1, 1)
        distance = (target_chunks.view(-1, 1) - source_chunks.view(1, -1)).clamp_min(0)
        distance[:, 0] = 0  # Never penalize the null item.
        # Keep deterministic masks in inference checkpoints. Transformers may
        # construct models on the meta device before loading weights, where
        # non-persistent buffers would otherwise remain uninitialized.
        self.register_buffer("memory_allowed", allowed, persistent=True)
        self.register_buffer("memory_distance", distance.float(), persistent=True)

        # summary_queries is not a Linear layer, so initialize it explicitly.
        nn.init.normal_(self.summary_queries, mean=0.0, std=0.02)

    def _build_summaries(self, x: torch.Tensor) -> torch.Tensor:
        batch, time, channels = x.shape
        chunk = self.chunk_size
        chunk_count = math.ceil(time / chunk)
        padded_time = chunk_count * chunk
        pad_tokens = padded_time - time

        if pad_tokens:
            x = F.pad(x, (0, 0, 0, pad_tokens))

        x_chunks = x.view(batch, chunk_count, chunk, channels)

        keys = self.summary_k(x_chunks)
        values = self.summary_v(x_chunks)

        # [B, C, S, W]
        scores = torch.einsum("bcwm,sm->bcsw", keys, self.summary_queries)
        scores = scores / math.sqrt(self.memory_dim)

        # Exclude padding from a partial final chunk.
        if pad_tokens:
            valid = torch.arange(padded_time, device=x.device) < time
            valid = valid.view(chunk_count, chunk)
            scores = scores.masked_fill(~valid.view(1, chunk_count, 1, chunk), float("-inf"))

        weights = F.softmax(scores.float(), dim=-1).to(dtype=x.dtype)
        summaries = torch.einsum("bcsw,bcwm->bcsm", weights, values)
        return summaries

    def forward(
        self,
        query_x: torch.Tensor,
        summary_x: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # query_x and summary_x are separate so memory can pool representations
        # that have already passed through local causal attention.
        if summary_x is None:
            summary_x = query_x

        batch, time, _ = query_x.shape
        if summary_x.shape[:2] != query_x.shape[:2]:
            raise ValueError("query_x and summary_x must have the same batch/time shape")

        chunk_count = math.ceil(time / self.chunk_size)

        summaries = self._build_summaries(summary_x)

        # Flatten summaries to [B, C*S, M].
        source = summaries.reshape(batch, chunk_count * self.summary_slots, self.memory_dim)
        null_memory = self.null_memory.expand(batch, -1, -1)
        source = torch.cat((null_memory, source), dim=1)

        q = self.read_q(query_x)
        k = self.read_k(source)
        v = self.read_v(source)

        q = q.view(batch, time, self.memory_heads, self.memory_head_dim).transpose(1, 2)
        k = k.view(batch, source.size(1), self.memory_heads, self.memory_head_dim).transpose(1, 2)
        v = v.view(batch, source.size(1), self.memory_heads, self.memory_head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.memory_head_dim)
        source_count = source.size(1)

        # Prefer recent chunks while still allowing content attention to override
        # the prior. softplus keeps every per-head decay non-negative.
        recency_decay = F.softplus(self.recency_decay_raw).view(1, self.memory_heads, 1, 1)
        distance = self.memory_distance[:time, :source_count].view(1, 1, time, source_count)
        scores = scores - recency_decay * distance

        allowed = self.memory_allowed[:time, :source_count]
        scores = scores.masked_fill(~allowed.view(1, 1, time, source_count), float("-inf"))

        weights = F.softmax(scores.float(), dim=-1).to(dtype=query_x.dtype)
        if self.training and self.dropout > 0.0:
            weights = F.dropout(weights, p=self.dropout)

        y = torch.matmul(weights, v)
        y = y.transpose(1, 2).contiguous().view(batch, time, self.memory_dim)
        return self.read_o(y)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.w1 = nn.Linear(cfg.d_model, cfg.ffn_dim, bias=False)
        self.w3 = nn.Linear(cfg.d_model, cfg.ffn_dim, bias=False)
        self.w2 = nn.Linear(cfg.ffn_dim, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class BindingBlock(nn.Module):
    """Independent final block for exact token-to-role and long-range binding."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()

        self.attn_norm = RMSNorm(cfg.d_model)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.attention = FullCausalAttention(cfg)
        self.ffn = SwiGLU(cfg)
        self.attn_gate = nn.Linear(cfg.d_model, 1, bias=True)
        self.ffn_gate = nn.Linear(cfg.d_model, 1, bias=True)
        self.residual_scale = 1.0 / math.sqrt(2.0)

        nn.init.zeros_(self.attn_gate.weight)
        nn.init.constant_(self.attn_gate.bias, cfg.residual_gate_init)
        nn.init.zeros_(self.ffn_gate.weight)
        nn.init.constant_(self.ffn_gate.bias, cfg.residual_gate_init)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> torch.Tensor:
        u = self.attn_norm(x)
        attention_update = self.attention(u, rope_cos, rope_sin)
        x = x + self.residual_scale * torch.sigmoid(self.attn_gate(u)) * attention_update

        v = self.ffn_norm(x)
        ffn_update = self.ffn(v)
        return x + self.residual_scale * torch.sigmoid(self.ffn_gate(v)) * ffn_update


class FoldedCell(nn.Module):
    """One physical recurrent cell whose parameters are reused across depths."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()

        self.attn_norm = RMSNorm(cfg.d_model)
        self.ffn_norm = RMSNorm(cfg.d_model)

        self.local_attention = LocalCausalAttention(cfg)
        self.summary_memory = CausalSummaryMemory(cfg)
        self.ffn = SwiGLU(cfg)

        # A scalar gate controls each token's residual update.
        self.attn_gate = nn.Linear(cfg.d_model, 1, bias=True)
        self.ffn_gate = nn.Linear(cfg.d_model, 1, bias=True)

        nn.init.zeros_(self.attn_gate.weight)
        nn.init.constant_(self.attn_gate.bias, cfg.residual_gate_init)
        nn.init.zeros_(self.ffn_gate.weight)
        nn.init.constant_(self.ffn_gate.bias, cfg.residual_gate_init)

    def forward(
        self,
        x: torch.Tensor,
        recurrence_index: int,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        attn_phase_scale: torch.Tensor,
        attn_phase_shift: torch.Tensor,
        ffn_phase_scale: torch.Tensor,
        ffn_phase_shift: torch.Tensor,
        memory_gain: torch.Tensor,
        residual_scale: float,
    ) -> torch.Tensor:
        # First sub-step: exact local path plus compressed long-range memory.
        u = phase_rms_norm(
            x,
            self.attn_norm,
            attn_phase_scale[recurrence_index],
            attn_phase_shift[recurrence_index],
        )

        local_update = self.local_attention(u, rope_cos, rope_sin)

        # Build causal local context before folding it into summaries. Those
        # summaries become visible only to later chunks.
        local_context = u + local_update
        memory_update = self.summary_memory(query_x=local_context, summary_x=local_context)
        memory_strength = torch.sigmoid(memory_gain[recurrence_index])

        mixed_update = local_update + memory_strength * memory_update
        attn_gate = torch.sigmoid(self.attn_gate(u))
        x = x + residual_scale * attn_gate * mixed_update

        # Second sub-step: shared SwiGLU.
        v = phase_rms_norm(
            x,
            self.ffn_norm,
            ffn_phase_scale[recurrence_index],
            ffn_phase_shift[recurrence_index],
        )
        ffn_update = self.ffn(v)
        ffn_gate = torch.sigmoid(self.ffn_gate(v))
        x = x + residual_scale * ffn_gate * ffn_update
        return x


class CFRDLanguageModel(nn.Module):
    """Causal Folded Recurrent Decoder."""

    def __init__(
        self,
        cfg: ModelConfig,
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

            self.register_buffer(
                "surface_feature_table",
                surface_feature_table.float(),
                # Inference exports need this tokenizer-derived table without
                # importing any project-specific tokenizer code.
                persistent=True,
            )
            self.surface_projection = nn.Linear(cfg.surface_feature_dim, cfg.d_model, bias=False)
            self.surface_gain = nn.Parameter(torch.tensor(cfg.surface_feature_gain_init, dtype=torch.float32))
        else:
            self.register_buffer("surface_feature_table", torch.empty(0), persistent=False)
            self.surface_projection = None
            self.surface_gain = None

        self.cells = nn.ModuleList([FoldedCell(cfg) for _ in range(cfg.physical_cells)])
        self.binding_block = BindingBlock(cfg) if cfg.use_binding_block else None

        # Small FiLM parameters let a reused cell specialize by recurrent depth.
        self.attn_phase_scale = nn.Parameter(torch.zeros(cfg.recurrences, cfg.d_model))
        self.attn_phase_shift = nn.Parameter(torch.zeros(cfg.recurrences, cfg.d_model))
        self.ffn_phase_scale = nn.Parameter(torch.zeros(cfg.recurrences, cfg.d_model))
        self.ffn_phase_shift = nn.Parameter(torch.zeros(cfg.recurrences, cfg.d_model))
        self.memory_gain = nn.Parameter(torch.full((cfg.recurrences,), cfg.memory_gain_init))

        self.final_norm = RMSNorm(cfg.d_model)

        head_dim = cfg.d_model // cfg.n_head
        rope_cos, rope_sin = build_rope_cache(cfg.context_length, head_dim, cfg.rope_theta)
        self.register_buffer("rope_cos", rope_cos, persistent=True)
        self.register_buffer("rope_sin", rope_sin, persistent=True)

        # Reused residual branches need a smaller update scale.
        self.residual_scale = 1.0 / math.sqrt(cfg.recurrences)

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
        # Start repeatedly visited residual output projections at a smaller scale.
        std = 0.02 / math.sqrt(2.0 * self.cfg.recurrences)

        for cell in self.cells:
            nn.init.normal_(cell.local_attention.o_proj.weight, mean=0.0, std=std)
            nn.init.normal_(cell.summary_memory.read_o.weight, mean=0.0, std=std)
            nn.init.normal_(cell.ffn.w2.weight, mean=0.0, std=std)

            # module.apply() touched these layers, so restore the intended gates.
            nn.init.zeros_(cell.attn_gate.weight)
            nn.init.constant_(cell.attn_gate.bias, self.cfg.residual_gate_init)
            nn.init.zeros_(cell.ffn_gate.weight)
            nn.init.constant_(cell.ffn_gate.bias, self.cfg.residual_gate_init)

        if self.binding_block is not None:
            nn.init.normal_(self.binding_block.attention.o_proj.weight, mean=0.0, std=std)
            nn.init.normal_(self.binding_block.ffn.w2.weight, mean=0.0, std=std)
            nn.init.zeros_(self.binding_block.attn_gate.weight)
            nn.init.constant_(self.binding_block.attn_gate.bias, self.cfg.residual_gate_init)
            nn.init.zeros_(self.binding_block.ffn_gate.weight)
            nn.init.constant_(self.binding_block.ffn_gate.bias, self.cfg.residual_gate_init)

    def _embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(token_ids)

        if self.cfg.use_surface_features:
            assert self.surface_projection is not None
            assert self.surface_gain is not None

            features = self.surface_feature_table[token_ids]
            surface = self.surface_projection(features.to(dtype=x.dtype))
            x = x + self.surface_gain.to(dtype=x.dtype) * surface

        return self.embedding_dropout(x)

    def _logits(self, x: torch.Tensor) -> torch.Tensor:
        # Tie the LM head to token embeddings to avoid a second vocabulary matrix.
        normalized = self.final_norm(x)
        return F.linear(normalized, self.token_embedding.weight)

    def forward(
        self,
        token_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        recurrences: int | None = None,
    ) -> ModelOutput:
        batch, time = token_ids.shape
        del batch

        if time > self.cfg.context_length:
            raise ValueError(f"Sequence length {time} exceeds context_length {self.cfg.context_length}")

        run_recurrences = self.cfg.recurrences if recurrences is None else recurrences
        if run_recurrences <= 0 or run_recurrences > self.cfg.recurrences:
            raise ValueError("recurrences must be between 1 and cfg.recurrences")

        x = self._embed(token_ids)

        exit_losses: dict[int, torch.Tensor] = {}
        final_logits: torch.Tensor | None = None

        for recurrence_index in range(run_recurrences):
            cell = self.cells[recurrence_index % len(self.cells)]
            x = cell(
                x=x,
                recurrence_index=recurrence_index,
                rope_cos=self.rope_cos,
                rope_sin=self.rope_sin,
                attn_phase_scale=self.attn_phase_scale,
                attn_phase_shift=self.attn_phase_shift,
                ffn_phase_scale=self.ffn_phase_scale,
                ffn_phase_shift=self.ffn_phase_shift,
                memory_gain=self.memory_gain,
                residual_scale=self.residual_scale,
            )

            depth = recurrence_index + 1
            should_project = depth == run_recurrences or (targets is not None and depth in self.cfg.exit_depths)

            if should_project:
                projection_x = x
                if self.binding_block is not None:
                    projection_x = self.binding_block(x, self.rope_cos, self.rope_sin)
                logits_at_depth = self._logits(projection_x)

                if depth == run_recurrences:
                    final_logits = logits_at_depth

                if targets is not None and depth in self.cfg.exit_depths:
                    exit_losses[depth] = F.cross_entropy(
                        logits_at_depth.reshape(-1, logits_at_depth.size(-1)),
                        targets.reshape(-1),
                    )

        assert final_logits is not None

        final_loss: torch.Tensor | None = None
        total_loss: torch.Tensor | None = None

        if targets is not None:
            if run_recurrences in exit_losses:
                final_loss = exit_losses[run_recurrences]
            else:
                final_loss = F.cross_entropy(
                    final_logits.reshape(-1, final_logits.size(-1)),
                    targets.reshape(-1),
                )

            auxiliary = [loss_value for depth, loss_value in exit_losses.items() if depth != run_recurrences]

            if auxiliary:
                aux_mean = torch.stack(auxiliary).mean()
                total_loss = final_loss + self.cfg.aux_exit_loss_weight * aux_mean
            else:
                total_loss = final_loss

        return ModelOutput(
            logits=final_logits,
            loss=total_loss,
            final_loss=final_loss,
            exit_losses=exit_losses,
        )


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"total": total, "trainable": trainable}
