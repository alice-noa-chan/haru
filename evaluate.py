from __future__ import annotations

import json

import numpy as np
import torch

import config
from model_factory import AnyLanguageModel, build_model, describe_architecture, model_config_from_checkpoint
from surface_features import build_surface_feature_table
from tokenizer_utils import StoryTokenizer
from train import (
    autocast_context,
    get_random_batch,
    load_packed_meta,
    open_token_stream,
    resolve_device,
    validate_packed_data,
)

FINAL_EVAL_SEED_OFFSET = 30_000


def make_final_eval_rng() -> np.random.Generator:
    """Recreate one shared validation-window sequence for every exit depth."""

    return np.random.default_rng(config.SEED + FINAL_EVAL_SEED_OFFSET)


@torch.inference_mode()
def evaluate_depth(
    model: AnyLanguageModel,
    data: np.memmap,
    device: torch.device,
    recurrence_depth: int,
) -> float:
    rng = make_final_eval_rng()
    total = 0.0

    for _ in range(config.FINAL_EVAL_BATCHES):
        x, y = get_random_batch(
            data,
            config.BATCH_SIZE,
            model.cfg.context_length,
            device,
            rng,
        )

        with autocast_context(device):
            output = model(x, targets=None, recurrences=recurrence_depth)
            loss = torch.nn.functional.cross_entropy(
                output.logits.reshape(-1, output.logits.size(-1)),
                y.reshape(-1),
            )

        total += float(loss.item())

    return total / config.FINAL_EVAL_BATCHES


def main() -> None:
    device = resolve_device()
    tokenizer = StoryTokenizer()
    meta = load_packed_meta()
    validate_packed_data(meta, tokenizer)

    checkpoint_path = (
        config.BEST_CHECKPOINT_PATH if config.BEST_CHECKPOINT_PATH.exists() else config.LATEST_CHECKPOINT_PATH
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError("No trained checkpoint was found")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    tokenizer.validate_checkpoint(checkpoint)
    model_cfg = model_config_from_checkpoint(checkpoint, tokenizer.vocab_size)
    surface_features = build_surface_feature_table(tokenizer)

    model = build_model(model_cfg, surface_features)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    print(f"Evaluating {describe_architecture(model_cfg)}", flush=True)

    val_data = open_token_stream(config.VAL_BIN_PATH)

    results: dict[str, float | int | str] = {}
    results["model_arch"] = describe_architecture(model_cfg).split()[0]
    results["checkpoint_step"] = int(checkpoint.get("step", 0))
    results["tokens_seen"] = int(checkpoint.get("tokens_seen", 0))
    results["evaluation_seed"] = config.SEED + FINAL_EVAL_SEED_OFFSET
    for depth in model_cfg.exit_depths:
        loss = evaluate_depth(model, val_data, device, depth)
        results[f"depth_{depth}_loss"] = loss
        results[f"depth_{depth}_perplexity"] = float(np.exp(min(loss, 20.0)))
        print(
            f"depth={depth} loss={loss:.5f} perplexity={results[f'depth_{depth}_perplexity']:.3f}",
            flush=True,
        )

    output_path = config.RUN_DIR / "evaluation.json"
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved evaluation: {output_path}", flush=True)


if __name__ == "__main__":
    main()
