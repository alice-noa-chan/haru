from __future__ import annotations

import json
import math
import random
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

import config
from data_utils import blake2b_file, dataset_fingerprint
from model import CFRDLanguageModel, ModelConfig, count_parameters
from surface_features import build_surface_feature_table
from tokenizer_utils import StoryTokenizer


def resolve_device() -> torch.device:
    """Resolve the configured device, preferring CUDA and then MPS."""

    if config.DEVICE != "auto":
        return torch.device(config.DEVICE)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_packed_meta() -> dict[str, Any]:
    if not config.PACKED_META_PATH.exists():
        raise FileNotFoundError(
            f"Packed metadata does not exist: {config.PACKED_META_PATH}\nRun `python prepare_data.py` first."
        )
    return json.loads(config.PACKED_META_PATH.read_text(encoding="utf-8"))


def validate_packed_data(meta: dict[str, Any], tokenizer: StoryTokenizer) -> None:
    """Reject stale or incompatible packed token streams."""

    for path in (config.TRAIN_BIN_PATH, config.VAL_BIN_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Packed token file does not exist: {path}")

    if int(meta["vocab_size"]) != tokenizer.vocab_size:
        raise ValueError("Packed data and tokenizer vocabulary sizes differ")

    expected_tokenizer_hash = str(meta["tokenizer"]["blake2b16"])
    current_tokenizer_hash = blake2b_file(config.TOKENIZER_MODEL_PATH)
    if expected_tokenizer_hash != current_tokenizer_hash:
        raise ValueError("The tokenizer changed after packing; run prepare_data.py again")

    expected_train_bytes = int(meta["train_tokens"]) * np.dtype(np.uint16).itemsize
    expected_val_bytes = int(meta["val_tokens"]) * np.dtype(np.uint16).itemsize

    if config.TRAIN_BIN_PATH.stat().st_size != expected_train_bytes:
        raise ValueError("train.bin size does not match meta.json")
    if config.VAL_BIN_PATH.stat().st_size != expected_val_bytes:
        raise ValueError("val.bin size does not match meta.json")

    if config.STRICT_DATA_FINGERPRINT:
        current_files = dataset_fingerprint()
        if current_files != meta.get("data_files"):
            raise ValueError("Source files changed after packing; run prepare_data.py again")


def open_token_stream(path: Path) -> np.memmap:
    return np.memmap(path, dtype=np.uint16, mode="r")


def get_random_batch(
    data: np.memmap,
    batch_size: int,
    context_length: int,
    device: torch.device,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample independent random windows from a packed token stream."""

    max_start = len(data) - context_length - 1
    if max_start <= 0:
        raise ValueError("Token stream is shorter than context_length + 1")

    starts = rng.integers(0, max_start + 1, size=batch_size, endpoint=False)

    # Copy memmap slices explicitly to avoid read-only tensor warnings.
    x_np = np.stack([np.array(data[start : start + context_length], dtype=np.int64, copy=True) for start in starts])
    y_np = np.stack(
        [np.array(data[start + 1 : start + context_length + 1], dtype=np.int64, copy=True) for start in starts]
    )

    x = torch.from_numpy(x_np).to(device=device, non_blocking=True)
    y = torch.from_numpy(y_np).to(device=device, non_blocking=True)
    return x, y


def learning_rate_for_step(step: int, max_steps: int) -> float:
    """Linear warmup followed by cosine decay."""

    if step < config.WARMUP_STEPS:
        return config.LEARNING_RATE * float(step + 1) / float(max(1, config.WARMUP_STEPS))

    if step >= max_steps:
        return config.MIN_LEARNING_RATE

    decay_steps = max(1, max_steps - config.WARMUP_STEPS)
    progress = (step - config.WARMUP_STEPS) / decay_steps
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.MIN_LEARNING_RATE + cosine * (config.LEARNING_RATE - config.MIN_LEARNING_RATE)


def configure_optimizer(model: torch.nn.Module, device: torch.device) -> torch.optim.Optimizer:
    """Apply weight decay to matrices, but not norms, biases, or scalar gates."""

    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []

    for _, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim >= 2:
            decay.append(parameter)
        else:
            no_decay.append(parameter)

    groups = [
        {"params": decay, "weight_decay": config.WEIGHT_DECAY},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    return torch.optim.AdamW(
        groups,
        lr=config.LEARNING_RATE,
        betas=(config.ADAM_BETA1, config.ADAM_BETA2),
        eps=config.ADAM_EPS,
        fused=(device.type == "cuda"),
    )


def autocast_context(device: torch.device):
    """Return an autocast context for the configured precision."""

    if config.PRECISION == "fp32" or device.type not in {"cuda", "cpu"}:
        return torch.autocast(device_type=device.type, enabled=False)

    if config.PRECISION == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)

    if config.PRECISION == "fp16":
        if device.type != "cuda":
            raise ValueError("fp16 training requires CUDA")
        return torch.autocast(device_type="cuda", dtype=torch.float16)

    raise ValueError(f"Unsupported PRECISION: {config.PRECISION}")


def optimizer_settings() -> dict[str, Any]:
    """Return settings that must remain unchanged when resuming a run."""

    return {
        "batch_size": config.BATCH_SIZE,
        "grad_accum_steps": config.GRAD_ACCUM_STEPS,
        "target_tokens": config.TARGET_TOKENS,
        "learning_rate": config.LEARNING_RATE,
        "min_learning_rate": config.MIN_LEARNING_RATE,
        "warmup_steps": config.WARMUP_STEPS,
        "weight_decay": config.WEIGHT_DECAY,
        "adam_beta1": config.ADAM_BETA1,
        "adam_beta2": config.ADAM_BETA2,
        "adam_eps": config.ADAM_EPS,
        "grad_clip_norm": config.GRAD_CLIP_NORM,
        "precision": config.PRECISION,
        "seed": config.SEED,
    }


def write_experiment_snapshot(model_cfg: ModelConfig, packed_meta: dict[str, Any]) -> None:
    """Write an immutable, human-readable description of a new run."""

    path = config.RUN_DIR / "experiment.json"
    if path.exists():
        return

    snapshot = {
        "run_name": config.RUN_NAME,
        "model_config": asdict(model_cfg),
        "training_config": optimizer_settings(),
        "packed_data": packed_meta,
    }
    temporary_path = path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_checkpoint(
    path: Path,
    model: CFRDLanguageModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    tokens_seen: int,
    best_val_loss: float,
    model_cfg: ModelConfig,
    tokenizer_hash: str,
    train_rng: np.random.Generator,
) -> None:
    """Atomically save all state required for an exact training resume."""

    state = {
        "checkpoint_version": 3,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "tokens_seen": tokens_seen,
        "best_val_loss": best_val_loss,
        "model_config": asdict(model_cfg),
        "optimizer_config": optimizer_settings(),
        "tokenizer_blake2b16": tokenizer_hash,
        "train_rng_state": train_rng.bit_generator.state,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }

    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def restore_checkpoint(
    path: Path,
    model: CFRDLanguageModel,
    optimizer: torch.optim.Optimizer,
    model_cfg: ModelConfig,
    tokenizer_hash: str,
    train_rng: np.random.Generator,
) -> tuple[int, int, float]:
    """Restore a checkpoint after validating its experiment configuration."""

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    saved_model_config = dict(checkpoint.get("model_config", {}))
    if saved_model_config.get("exit_depths") is not None:
        saved_model_config["exit_depths"] = tuple(saved_model_config["exit_depths"])
    if saved_model_config != asdict(model_cfg):
        raise ValueError(
            "Checkpoint model configuration differs from config.py. Use a new RUN_NAME for a new architecture."
        )

    if checkpoint.get("optimizer_config") != optimizer_settings():
        raise ValueError(
            "Checkpoint training settings differ from config.py. "
            "Use a new RUN_NAME instead of silently mixing experiments."
        )

    if checkpoint.get("tokenizer_blake2b16") != tokenizer_hash:
        raise ValueError("Checkpoint was trained with a different tokenizer")

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    train_rng_state = checkpoint.get("train_rng_state")
    if train_rng_state is None:
        raise ValueError("Checkpoint does not contain the exact training sampler state")
    train_rng.bit_generator.state = train_rng_state

    rng = checkpoint.get("rng")
    if rng:
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        if torch.cuda.is_available() and rng.get("cuda") is not None:
            torch.cuda.set_rng_state_all(rng["cuda"])

    return (
        int(checkpoint.get("step", 0)),
        int(checkpoint.get("tokens_seen", 0)),
        float(checkpoint.get("best_val_loss", float("inf"))),
    )


@torch.no_grad()
def evaluate(
    model: CFRDLanguageModel,
    val_data: np.memmap,
    device: torch.device,
) -> dict[str, float]:
    """Use fixed validation windows so checkpoint comparisons are stable."""

    model.eval()
    rng = np.random.default_rng(config.SEED + 10_000)

    total_loss = 0.0
    total_final_loss = 0.0
    total_exit_losses: dict[int, float] = {depth: 0.0 for depth in model.cfg.exit_depths}

    for _ in range(config.EVAL_BATCHES):
        x, y = get_random_batch(
            val_data,
            config.BATCH_SIZE,
            config.CONTEXT_LENGTH,
            device,
            rng,
        )

        with autocast_context(device):
            output = model(x, targets=y)

        assert output.loss is not None
        assert output.final_loss is not None

        total_loss += float(output.loss.item())
        total_final_loss += float(output.final_loss.item())

        for depth, loss_value in output.exit_losses.items():
            total_exit_losses[depth] += float(loss_value.item())

    result = {
        "loss": total_loss / config.EVAL_BATCHES,
        "final_loss": total_final_loss / config.EVAL_BATCHES,
    }

    for depth, value in total_exit_losses.items():
        result[f"exit_{depth}_loss"] = value / config.EVAL_BATCHES

    model.train()
    return result


def main() -> None:
    config.RUN_DIR.mkdir(parents=True, exist_ok=True)
    seed_everything(config.SEED)

    device = resolve_device()
    torch.set_float32_matmul_precision("high")

    tokenizer = StoryTokenizer()
    meta = load_packed_meta()
    validate_packed_data(meta, tokenizer)
    tokenizer_hash = blake2b_file(config.TOKENIZER_MODEL_PATH)

    surface_features = build_surface_feature_table(tokenizer)
    model_cfg = ModelConfig.from_project_settings(config, tokenizer.vocab_size)
    model = CFRDLanguageModel(model_cfg, surface_features)
    parameter_count = count_parameters(model)
    write_experiment_snapshot(model_cfg, meta)

    model.to(device)

    optimizer = configure_optimizer(model, device)

    start_step = 0
    tokens_seen = 0
    best_val_loss = float("inf")
    train_rng = np.random.default_rng(config.SEED)

    if config.LATEST_CHECKPOINT_PATH.exists():
        print(f"Resuming checkpoint: {config.LATEST_CHECKPOINT_PATH}", flush=True)
        start_step, tokens_seen, best_val_loss = restore_checkpoint(
            config.LATEST_CHECKPOINT_PATH,
            model,
            optimizer,
            model_cfg,
            tokenizer_hash,
            train_rng,
        )
        model.to(device)

    train_model: torch.nn.Module = model
    if config.USE_TORCH_COMPILE:
        train_model = torch.compile(model)

    train_data = open_token_stream(config.TRAIN_BIN_PATH)
    val_data = open_token_stream(config.VAL_BIN_PATH)

    tokens_per_optimizer_step = config.BATCH_SIZE * config.CONTEXT_LENGTH * config.GRAD_ACCUM_STEPS
    max_steps = math.ceil(config.TARGET_TOKENS / tokens_per_optimizer_step)

    print("=" * 78, flush=True)
    print("CFRD training", flush=True)
    print(f"device={device}", flush=True)
    print(f"precision={config.PRECISION}", flush=True)
    print(f"parameters={parameter_count['total']:,}", flush=True)
    print(f"train_tokens={len(train_data):,}", flush=True)
    print(f"val_tokens={len(val_data):,}", flush=True)
    print(f"tokens_per_optimizer_step={tokens_per_optimizer_step:,}", flush=True)
    print(f"target_tokens={config.TARGET_TOKENS:,}", flush=True)
    print(f"max_steps={max_steps:,}", flush=True)
    print("=" * 78, flush=True)

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(config.PRECISION == "fp16" and device.type == "cuda"),
    )

    model.train()
    wall_start = time.perf_counter()

    for step in range(start_step, max_steps):
        step_start = time.perf_counter()

        lr = learning_rate_for_step(step, max_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        accumulated_final_loss = 0.0

        for micro_step in range(config.GRAD_ACCUM_STEPS):
            x, y = get_random_batch(
                train_data,
                config.BATCH_SIZE,
                config.CONTEXT_LENGTH,
                device,
                train_rng,
            )

            sync_context = (
                train_model.no_sync()
                if hasattr(train_model, "no_sync") and micro_step < config.GRAD_ACCUM_STEPS - 1
                else nullcontext()
            )

            with sync_context:
                with autocast_context(device):
                    output = train_model(x, targets=y)

                assert output.loss is not None
                assert output.final_loss is not None
                loss = output.loss / config.GRAD_ACCUM_STEPS

                accumulated_loss += float(output.loss.item()) / config.GRAD_ACCUM_STEPS
                accumulated_final_loss += float(output.final_loss.item()) / config.GRAD_ACCUM_STEPS

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

        if scaler.is_enabled():
            scaler.unscale_(optimizer)

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)

        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        step_tokens = tokens_per_optimizer_step
        tokens_seen += step_tokens
        elapsed = time.perf_counter() - step_start
        tokens_per_second = step_tokens / max(elapsed, 1.0e-9)

        current_step = step + 1

        if current_step % config.LOG_INTERVAL == 0 or current_step == 1:
            row = {
                "step": current_step,
                "max_steps": max_steps,
                "tokens_seen": tokens_seen,
                "loss": accumulated_loss,
                "final_loss": accumulated_final_loss,
                "lr": lr,
                "grad_norm": float(grad_norm),
                "tokens_per_second": tokens_per_second,
                "elapsed_seconds": time.perf_counter() - wall_start,
            }
            append_jsonl(config.TRAIN_LOG_PATH, row)
            print(
                f"step={current_step:6d}/{max_steps} "
                f"tokens={tokens_seen:,} "
                f"loss={accumulated_loss:.4f} "
                f"final={accumulated_final_loss:.4f} "
                f"lr={lr:.2e} "
                f"tok/s={tokens_per_second:,.0f}",
                flush=True,
            )

        should_eval = current_step % config.EVAL_INTERVAL == 0 or current_step == max_steps
        if should_eval:
            metrics = evaluate(model, val_data, device)
            append_jsonl(
                config.TRAIN_LOG_PATH,
                {
                    "step": current_step,
                    "tokens_seen": tokens_seen,
                    "validation": metrics,
                },
            )

            print(
                f"validation step={current_step} loss={metrics['loss']:.4f} final={metrics['final_loss']:.4f}",
                flush=True,
            )

            if metrics["final_loss"] < best_val_loss:
                best_val_loss = metrics["final_loss"]
                save_checkpoint(
                    config.BEST_CHECKPOINT_PATH,
                    model,
                    optimizer,
                    current_step,
                    tokens_seen,
                    best_val_loss,
                    model_cfg,
                    tokenizer_hash,
                    train_rng,
                )
                print(f"Saved best checkpoint: {config.BEST_CHECKPOINT_PATH}", flush=True)

        should_save = current_step % config.SAVE_INTERVAL == 0 or current_step == max_steps
        if should_save:
            save_checkpoint(
                config.LATEST_CHECKPOINT_PATH,
                model,
                optimizer,
                current_step,
                tokens_seen,
                best_val_loss,
                model_cfg,
                tokenizer_hash,
                train_rng,
            )
            print(f"Saved latest checkpoint: {config.LATEST_CHECKPOINT_PATH}", flush=True)

    print("Training complete", flush=True)


if __name__ == "__main__":
    main()
