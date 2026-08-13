"""Select between the CFRD architecture and its dense Transformer control.

Training, evaluation, and export all need to build "the model for this run"
without hard-coding CFRD. Keeping the dispatch here means an ablation run
differs from a release run by one setting in config.py rather than by an edited
copy of train.py, which is how ablation results drift away from the code that
produced them.

Checkpoints written before this module existed carry no architecture tag. They
are CFRD by construction, so they are still readable.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Union

import torch

from baseline_model import BaselineConfig, BaselineLanguageModel
from model import CFRDLanguageModel, ModelConfig

CFRD_ARCH = "cfrd"
BASELINE_ARCH = "dense-baseline"
SUPPORTED_ARCHITECTURES = (CFRD_ARCH, BASELINE_ARCH)

AnyModelConfig = Union[ModelConfig, BaselineConfig]
AnyLanguageModel = Union[CFRDLanguageModel, BaselineLanguageModel]


def normalize_architecture(name: str) -> str:
    architecture = str(name).strip().lower()
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(f"Unknown architecture {name!r}. Choose one of: {', '.join(SUPPORTED_ARCHITECTURES)}")
    return architecture


def architecture_of_config(model_cfg: AnyModelConfig) -> str:
    if isinstance(model_cfg, ModelConfig):
        return CFRD_ARCH
    if isinstance(model_cfg, BaselineConfig):
        return BASELINE_ARCH
    raise TypeError(f"Unsupported model configuration type: {type(model_cfg).__name__}")


def architecture_of_checkpoint(checkpoint: dict) -> str:
    """Read a checkpoint's architecture, defaulting to CFRD for older files.

    Haru v1.0 and v1.1 checkpoints predate the tag. Inferring from the stored
    fields keeps them loadable without rewriting released artifacts.
    """

    tagged = checkpoint.get("model_arch")
    if tagged is not None:
        return normalize_architecture(tagged)

    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("Checkpoint does not contain model_config")
    return BASELINE_ARCH if "n_layer" in model_config else CFRD_ARCH


def build_model_config(settings: object, vocab_size: int, architecture: str | None = None) -> AnyModelConfig:
    """Build the configured architecture's config from config.py settings."""

    resolved = normalize_architecture(architecture or getattr(settings, "MODEL_ARCH", CFRD_ARCH))
    if resolved == BASELINE_ARCH:
        return BaselineConfig.from_project_settings(settings, vocab_size)
    return ModelConfig.from_project_settings(settings, vocab_size)


def model_config_from_checkpoint(checkpoint: dict, vocab_size: int) -> AnyModelConfig:
    """Rebuild the exact configuration a checkpoint was trained with."""

    if architecture_of_checkpoint(checkpoint) == BASELINE_ARCH:
        return BaselineConfig.from_checkpoint(checkpoint, vocab_size)
    return ModelConfig.from_checkpoint(checkpoint, vocab_size)


def build_model(model_cfg: AnyModelConfig, surface_feature_table: torch.Tensor | None) -> AnyLanguageModel:
    """Instantiate the model matching a configuration object."""

    if architecture_of_config(model_cfg) == BASELINE_ARCH:
        assert isinstance(model_cfg, BaselineConfig)
        return BaselineLanguageModel(model_cfg, surface_feature_table)

    assert isinstance(model_cfg, ModelConfig)
    return CFRDLanguageModel(model_cfg, surface_feature_table)


def describe_architecture(model_cfg: AnyModelConfig) -> str:
    """One-line shape summary for training logs and experiment snapshots."""

    architecture = architecture_of_config(model_cfg)
    if architecture == BASELINE_ARCH:
        assert isinstance(model_cfg, BaselineConfig)
        return f"{architecture} layers={model_cfg.n_layer} d_model={model_cfg.d_model} ffn={model_cfg.ffn_dim}"

    assert isinstance(model_cfg, ModelConfig)
    return (
        f"{architecture} cells={model_cfg.physical_cells} recurrences={model_cfg.recurrences} "
        f"d_model={model_cfg.d_model} ffn={model_cfg.ffn_dim} binding_block={model_cfg.use_binding_block}"
    )


def config_as_dict(model_cfg: AnyModelConfig) -> dict:
    """Serialize a model configuration for checkpoints and snapshots."""

    return asdict(model_cfg)
