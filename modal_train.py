"""Benchmark, prepare, and train Haru on Modal GPUs."""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import modal


APP_NAME = "haru"
VOLUME_NAME = "haru-training"
REMOTE_PROJECT_DIR = "/root/haru"
REMOTE_STORAGE_DIR = "/vol"

LOCAL_PROJECT_DIR = Path(__file__).resolve().parent
SOURCE_FILES = sorted(LOCAL_PROJECT_DIR.glob("*.py"))

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch==2.12.1",
        "numpy==2.5.0",
        "sentencepiece==0.2.1",
        "transformers==5.12.1",
        "safetensors==0.8.0",
    )
    .env({"HARU_STORAGE_DIR": REMOTE_STORAGE_DIR})
)
for source_file in SOURCE_FILES:
    image = image.add_local_file(
        source_file,
        remote_path=f"{REMOTE_PROJECT_DIR}/{source_file.name}",
    )

app = modal.App(APP_NAME, image=image)
training_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _enter_project() -> None:
    os.chdir(REMOTE_PROJECT_DIR)
    # Modal imports this file from /root, while the rest of the project is
    # mounted below /root/haru. Add that directory explicitly so ordinary
    # imports such as `import config` work in every remote function.
    if REMOTE_PROJECT_DIR not in sys.path:
        sys.path.insert(0, REMOTE_PROJECT_DIR)


def _configure_gpu_training(gpu_name: str, batch_size: int, target_tokens: int) -> None:
    """Apply cloud-specific runtime settings before importing the training loop."""

    import config

    config.DEVICE = "cuda"
    config.PRECISION = "fp16" if gpu_name == "T4" else "bf16"
    config.BATCH_SIZE = batch_size

    # Keep the original 131,072-token optimizer step when the micro-batch changes.
    original_step_tokens = 32 * config.CONTEXT_LENGTH * 8
    micro_batch_tokens = batch_size * config.CONTEXT_LENGTH
    config.GRAD_ACCUM_STEPS = max(1, math.ceil(original_step_tokens / micro_batch_tokens))
    config.TARGET_TOKENS = target_tokens
    config.SAVE_INTERVAL = min(config.SAVE_INTERVAL, 100)
    config.USE_TORCH_COMPILE = False


def _run_benchmark(gpu_name: str, precision: str, batch_sizes: tuple[int, ...]) -> list[dict]:
    """Measure real forward/backward throughput for the complete CFRD objective."""

    _enter_project()

    import torch

    import config
    from model import CFRDLanguageModel, ModelConfig
    from surface_features import SURFACE_FEATURE_DIM
    from train import autocast_context, configure_optimizer

    config.DEVICE = "cuda"
    config.PRECISION = precision
    config.USE_TORCH_COMPILE = False
    torch.set_float32_matmul_precision("high")

    hourly_prices = {"T4": 0.000164 * 3600.0, "L4": 0.000222 * 3600.0}
    results: list[dict] = []

    for batch_size in batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            model_config = ModelConfig.from_project_settings(config, config.TOKENIZER_VOCAB_SIZE)
            surface_features = torch.zeros(
                config.TOKENIZER_VOCAB_SIZE,
                SURFACE_FEATURE_DIM,
                dtype=torch.float32,
            )
            model = CFRDLanguageModel(model_config, surface_features).cuda().train()
            optimizer = configure_optimizer(model, torch.device("cuda"))
            scaler = torch.amp.GradScaler("cuda", enabled=(precision == "fp16"))
            input_ids = torch.randint(
                0,
                config.TOKENIZER_VOCAB_SIZE,
                (batch_size, config.CONTEXT_LENGTH),
                device="cuda",
            )
            targets = torch.randint(
                0,
                config.TOKENIZER_VOCAB_SIZE,
                input_ids.shape,
                device="cuda",
            )

            warmup_steps = 2
            measured_steps = 5
            elapsed = 0.0

            for step in range(warmup_steps + measured_steps):
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                started = time.perf_counter()
                with autocast_context(torch.device("cuda")):
                    output = model(input_ids, targets=targets)
                assert output.loss is not None
                if scaler.is_enabled():
                    scaler.scale(output.loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    output.loss.backward()
                    optimizer.step()
                torch.cuda.synchronize()
                if step >= warmup_steps:
                    elapsed += time.perf_counter() - started

            tokens = measured_steps * batch_size * config.CONTEXT_LENGTH
            tokens_per_second = tokens / elapsed
            cost_per_billion_tokens = 1_000_000_000 / tokens_per_second / 3600.0 * hourly_prices[gpu_name]
            results.append(
                {
                    "gpu": gpu_name,
                    "precision": precision,
                    "batch_size": batch_size,
                    "tokens_per_second": round(tokens_per_second, 1),
                    "peak_memory_gib": round(torch.cuda.max_memory_allocated() / (1024**3), 2),
                    "estimated_hours_800m": round(800_000_000 / tokens_per_second / 3600.0, 2),
                    "estimated_gpu_cost_800m_usd": round(
                        800_000_000 / tokens_per_second / 3600.0 * hourly_prices[gpu_name],
                        2,
                    ),
                    "estimated_gpu_cost_1b_usd": round(cost_per_billion_tokens, 2),
                }
            )
            del model, optimizer, scaler, input_ids, targets, output
        except torch.OutOfMemoryError:
            results.append(
                {
                    "gpu": gpu_name,
                    "precision": precision,
                    "batch_size": batch_size,
                    "error": "CUDA out of memory",
                }
            )
            torch.cuda.empty_cache()

    print(json.dumps(results, indent=2), flush=True)
    return results


@app.function(gpu="T4", timeout=20 * 60)
def benchmark_t4(batch_sizes: tuple[int, ...] = (8, 16, 32)) -> list[dict]:
    return _run_benchmark("T4", "fp16", batch_sizes)


@app.function(gpu="L4", timeout=20 * 60)
def benchmark_l4(batch_sizes: tuple[int, ...] = (8, 16, 32)) -> list[dict]:
    return _run_benchmark("L4", "bf16", batch_sizes)


@app.function(
    cpu=8,
    memory=16_384,
    timeout=24 * 60 * 60,
    volumes={REMOTE_STORAGE_DIR: training_volume},
)
def prepare_remote_data() -> dict[str, str]:
    """Train the tokenizer and create packed train/validation streams once."""

    _enter_project()
    import config
    import prepare_data
    import tokenizer_train

    if not config.TOKENIZER_MODEL_PATH.exists():
        tokenizer_train.main()
    else:
        print(f"Tokenizer already exists: {config.TOKENIZER_MODEL_PATH}", flush=True)

    if not config.PACKED_META_PATH.exists():
        prepare_data.main()
    else:
        print(f"Packed data already exists: {config.PACKED_META_PATH}", flush=True)

    training_volume.commit()
    return {
        "tokenizer": str(config.TOKENIZER_MODEL_PATH),
        "packed_meta": str(config.PACKED_META_PATH),
    }


def _train(gpu_name: str, batch_size: int, target_tokens: int) -> dict[str, str]:
    _enter_project()
    _configure_gpu_training(gpu_name, batch_size, target_tokens)

    import config
    import train

    if not config.PACKED_META_PATH.exists():
        raise FileNotFoundError("Run the Modal prepare action before training")

    # Persist every checkpoint while the function is still running. Modal also
    # commits the Volume when the function exits normally, but committing here
    # protects recent progress if a long GPU run is interrupted.
    original_save_checkpoint = train.save_checkpoint

    def save_checkpoint_and_commit(*args, **kwargs) -> None:
        original_save_checkpoint(*args, **kwargs)
        training_volume.commit()

    train.save_checkpoint = save_checkpoint_and_commit

    train.main()
    training_volume.commit()
    return {
        "run_dir": str(config.RUN_DIR),
        "latest_checkpoint": str(config.LATEST_CHECKPOINT_PATH),
        "best_checkpoint": str(config.BEST_CHECKPOINT_PATH),
    }


@app.function(
    gpu="T4",
    timeout=24 * 60 * 60,
    volumes={REMOTE_STORAGE_DIR: training_volume},
)
def train_t4(batch_size: int = 16, target_tokens: int = 800_000_000) -> dict[str, str]:
    return _train("T4", batch_size, target_tokens)


@app.function(
    gpu="L4",
    timeout=24 * 60 * 60,
    volumes={REMOTE_STORAGE_DIR: training_volume},
)
def train_l4(batch_size: int = 16, target_tokens: int = 800_000_000) -> dict[str, str]:
    return _train("L4", batch_size, target_tokens)


@app.function(
    gpu="L4",
    timeout=60 * 60,
    volumes={REMOTE_STORAGE_DIR: training_volume},
)
def evaluate_l4() -> str:
    """Run the full held-out evaluation at every supervised recurrence depth."""

    _enter_project()
    import config
    import evaluate

    config.DEVICE = "cuda"
    config.PRECISION = "bf16"
    config.BATCH_SIZE = 32
    evaluate.main()
    training_volume.commit()
    return str(config.RUN_DIR / "evaluation.json")


@app.function(
    cpu=4,
    memory=8192,
    timeout=30 * 60,
    volumes={REMOTE_STORAGE_DIR: training_volume},
)
def export_remote() -> str:
    """Export and reload the best checkpoint through Transformers AutoClass."""

    _enter_project()
    import config
    import export_transformers

    export_transformers.main()
    training_volume.commit()
    return str(config.TRANSFORMERS_EXPORT_DIR)


@app.local_entrypoint()
def main(
    action: str = "benchmark",
    gpu: str = "L4",
    batch_size: int = 16,
    target_tokens: int = 800_000_000,
) -> None:
    """Run a benchmark, data, training, evaluation, or export action."""

    normalized_action = action.lower()
    normalized_gpu = gpu.upper()

    if normalized_action == "benchmark":
        t4_call = benchmark_t4.spawn()
        l4_call = benchmark_l4.spawn()
        results = {
            "T4": t4_call.get(),
            "L4": l4_call.get(),
        }
        print(json.dumps(results, indent=2))
        return

    if normalized_action == "upload":
        local_data_dir = LOCAL_PROJECT_DIR / "data"
        if not local_data_dir.exists():
            raise FileNotFoundError(f"Local data directory does not exist: {local_data_dir}")
        with training_volume.batch_upload() as upload:
            upload.put_directory(local_data_dir, "/data")
        print(f"Uploaded {local_data_dir} to {VOLUME_NAME}:/data")
        return

    if normalized_action == "prepare":
        print(prepare_remote_data.remote())
        return

    if normalized_action == "train":
        if normalized_gpu == "T4":
            print(train_t4.remote(batch_size, target_tokens))
        elif normalized_gpu == "L4":
            print(train_l4.remote(batch_size, target_tokens))
        else:
            raise ValueError("gpu must be T4 or L4")
        return

    if normalized_action == "evaluate":
        print(evaluate_l4.remote())
        return

    if normalized_action == "export":
        print(export_remote.remote())
        return

    raise ValueError("action must be benchmark, upload, prepare, train, evaluate, or export")
