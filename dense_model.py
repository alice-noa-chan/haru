"""Haru v3 decoder: full attention, QK normalization and an incremental KV cache."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class DenseConfig:
    vocab_size: int = 12_000
    context_length: int = 1024
    d_model: int = 384
    n_head: int = 6
    n_kv_head: int = 2
    ffn_dim: int = 1024
    n_layer: int = 8
    rope_theta: float = 10_000.0
    dropout: float = 0.0
    attention_gate: bool = False
    use_surface_features: bool = True
    surface_feature_dim: int = 76
    surface_feature_gain_init: float = 0.1

    @property
    def exit_depths(self):
        return (self.n_layer,)

    def validate(self):
        for key in ("vocab_size", "context_length", "d_model", "n_head", "n_kv_head", "ffn_dim", "n_layer"):
            if getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive")
        if self.d_model % self.n_head or self.n_head % self.n_kv_head:
            raise ValueError("Invalid head dimensions or GQA grouping")
        if (self.d_model // self.n_head) % 2:
            raise ValueError("RoPE head dimension must be even")
        if not 0 <= self.dropout < 1:
            raise ValueError("Invalid dropout")

    @classmethod
    def from_checkpoint(cls, checkpoint, vocab_size):
        return cls(**{**checkpoint["model_config"], "vocab_size": vocab_size})


CANDIDATES = {
    "dense8": DenseConfig(),
    "deep10": DenseConfig(d_model=320, n_head=5, n_kv_head=1, ffn_dim=1152, n_layer=10),
    "gated8": DenseConfig(ffn_dim=960, attention_gate=True),
}


class DenseRMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        return normalized.to(x.dtype) * self.weight.to(x.dtype)


def rotate(x, positions, theta):
    width = x.shape[-1]
    frequency = theta ** (-torch.arange(0, width, 2, device=x.device, dtype=torch.float32) / width)
    angle = positions.float()[..., None] * frequency
    angle = torch.cat((angle, angle), -1).unsqueeze(1)
    first, second = x.chunk(2, -1)
    rotated = torch.cat((-second, first), -1)
    return x * angle.cos().to(x.dtype) + rotated * angle.sin().to(x.dtype)


class DenseAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.head_dim = cfg.d_model // cfg.n_head
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.q_norm = DenseRMSNorm(self.head_dim)
        self.k_norm = DenseRMSNorm(self.head_dim)
        self.gate = nn.Linear(cfg.d_model, cfg.d_model, bias=False) if cfg.attention_gate else None

    def forward(self, x, positions, mask, past, use_cache):
        batch, length, _ = x.shape
        q = self.q_proj(x).view(batch, length, self.cfg.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, length, self.cfg.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, self.cfg.n_kv_head, self.head_dim).transpose(1, 2)
        q = rotate(self.q_norm(q), positions, self.cfg.rope_theta)
        k = rotate(self.k_norm(k), positions, self.cfg.rope_theta)
        past_length = 0 if past is None else past[0].shape[2]
        if past is not None:
            k, v = torch.cat((past[0], k), 2), torch.cat((past[1], v), 2)
        present = (k, v) if use_cache else None
        # Explicit repetition works on CPU, CUDA and export runtimes without an enable_gqa dependency.
        repeats = self.cfg.n_head // self.cfg.n_kv_head
        keys, values = k.repeat_interleave(repeats, 1), v.repeat_interleave(repeats, 1)
        causal = past is None and mask is None
        attention_mask = None
        if not causal:
            queries = past_length + torch.arange(length, device=x.device)
            allowed = torch.arange(k.shape[2], device=x.device)[None, :] <= queries[:, None]
            attention_mask = allowed[None, None, :, :]
            if mask is not None:
                attention_mask = attention_mask & mask[:, None, None, :].bool()
        y = F.scaled_dot_product_attention(
            q,
            keys,
            values,
            attn_mask=attention_mask,
            is_causal=causal,
            dropout_p=self.cfg.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(batch, length, self.cfg.d_model)
        if self.gate is not None:
            y = y * torch.sigmoid(self.gate(x))
        return self.o_proj(y), present


class DenseFFN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.w1 = nn.Linear(cfg.d_model, cfg.ffn_dim, bias=False)
        self.w3 = nn.Linear(cfg.d_model, cfg.ffn_dim, bias=False)
        self.w2 = nn.Linear(cfg.ffn_dim, cfg.d_model, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class DenseBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn_norm, self.ffn_norm = DenseRMSNorm(cfg.d_model), DenseRMSNorm(cfg.d_model)
        self.attention, self.ffn = DenseAttention(cfg), DenseFFN(cfg)

    def forward(self, x, positions, mask, past, use_cache):
        update, present = self.attention(self.attn_norm(x), positions, mask, past, use_cache)
        x = x + update
        return x + self.ffn(self.ffn_norm(x)), present


@dataclass
class DenseOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    final_loss: torch.Tensor | None = None
    exit_losses: dict | None = None
    past_key_values: tuple | None = None


class DenseLanguageModel(nn.Module):
    def __init__(self, cfg, surface_feature_table=None):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        if cfg.use_surface_features:
            if surface_feature_table is None:
                raise ValueError("surface_feature_table is required")
            if tuple(surface_feature_table.shape) != (cfg.vocab_size, cfg.surface_feature_dim):
                raise ValueError("Invalid surface feature shape")
            self.register_buffer("surface_feature_table", surface_feature_table.float())
            self.surface_projection = nn.Linear(cfg.surface_feature_dim, cfg.d_model, bias=False)
            self.surface_gain = nn.Parameter(torch.tensor(cfg.surface_feature_gain_init))
        else:
            self.register_buffer("surface_feature_table", torch.empty(0), persistent=False)
            self.surface_projection = None
        self.blocks = nn.ModuleList([DenseBlock(cfg) for _ in range(cfg.n_layer)])
        self.final_norm = DenseRMSNorm(cfg.d_model)
        self.apply(self._initialize)
        for block in self.blocks:
            for output in (block.attention.o_proj, block.ffn.w2):
                nn.init.normal_(output.weight, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        token_ids,
        targets=None,
        recurrences=None,
        logits_to_keep=0,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
    ):
        batch, length = token_ids.shape
        if length == 0:
            raise ValueError("Empty input")
        if recurrences is not None and recurrences != self.cfg.n_layer:
            raise ValueError("Haru dense has no recurrent depth setting")
        if logits_to_keep < 0 or (targets is not None and logits_to_keep):
            raise ValueError("Invalid logits_to_keep")
        past_length = 0 if past_key_values is None else past_key_values[0][0].shape[2]
        if past_length + length > self.cfg.context_length:
            raise ValueError("Input and cache exceed context_length")
        if past_key_values is not None and len(past_key_values) != self.cfg.n_layer:
            raise ValueError("Cache layer count differs")
        if attention_mask is not None:
            if tuple(attention_mask.shape) != (batch, past_length + length):
                raise ValueError("attention_mask must include cached and new tokens")
            if not bool(attention_mask.bool().any(-1).all()):
                raise ValueError("Every row needs a non-padding token")
        if position_ids is None:
            if attention_mask is None:
                position_ids = torch.arange(past_length, past_length + length, device=token_ids.device)[None, :]
            else:
                position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)[:, -length:]
        x = self.token_embedding(token_ids)
        if self.surface_projection is not None:
            x = x + self.surface_gain.to(x.dtype) * self.surface_projection(
                self.surface_feature_table[token_ids].to(x.dtype)
            )
        presents = []
        for index, block in enumerate(self.blocks):
            x, present = block(
                x, position_ids, attention_mask, None if past_key_values is None else past_key_values[index], use_cache
            )
            if use_cache:
                presents.append(present)
        if logits_to_keep:
            x = x[:, -logits_to_keep:]
        logits = F.linear(self.final_norm(x), self.token_embedding.weight)
        loss = (
            None if targets is None else F.cross_entropy(logits.reshape(-1, self.cfg.vocab_size), targets.reshape(-1))
        )
        return DenseOutput(
            logits,
            loss,
            loss,
            {self.cfg.n_layer: loss} if loss is not None else {},
            tuple(presents) if use_cache else None,
        )


def parameter_count(cfg):
    head = cfg.d_model // cfg.n_head
    attention = 2 * cfg.d_model**2 + 2 * cfg.d_model * cfg.n_kv_head * head + 2 * head
    gate = cfg.d_model**2 if cfg.attention_gate else 0
    surface = cfg.surface_feature_dim * cfg.d_model + 1 if cfg.use_surface_features else 0
    return (
        cfg.vocab_size * cfg.d_model
        + surface
        + cfg.d_model
        + cfg.n_layer * (attention + gate + 3 * cfg.d_model * cfg.ffn_dim + 2 * cfg.d_model)
    )


def grow_teacher(student):
    cfg = replace(student.cfg, n_layer=2 * student.cfg.n_layer)
    teacher = DenseLanguageModel(cfg, student.surface_feature_table).to(student.token_embedding.weight.device)
    old = student.state_dict()
    teacher.load_state_dict({key: value for key, value in old.items() if not key.startswith("blocks.")}, strict=False)
    for index, block in enumerate(student.blocks):
        teacher.blocks[2 * index].load_state_dict(block.state_dict())
        nn.init.zeros_(teacher.blocks[2 * index + 1].attention.o_proj.weight)
        nn.init.zeros_(teacher.blocks[2 * index + 1].ffn.w2.weight)
    return teacher


def student_from_teacher(teacher):
    if teacher.cfg.n_layer % 2:
        raise ValueError("Teacher depth must be even")
    cfg = replace(teacher.cfg, n_layer=teacher.cfg.n_layer // 2)
    student = DenseLanguageModel(cfg, teacher.surface_feature_table).to(teacher.token_embedding.weight.device)
    student.load_state_dict(
        {key: value for key, value in teacher.state_dict().items() if not key.startswith("blocks.")}, strict=False
    )
    for index, block in enumerate(student.blocks):
        block.load_state_dict(teacher.blocks[2 * index].state_dict())
    return student


def dense_checkpoint_config(model):
    return asdict(model.cfg)
