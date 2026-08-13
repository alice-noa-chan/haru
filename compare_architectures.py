"""Train CFRD against parameter- and compute-matched dense baselines.

README lists "no parameter-matched Transformer baseline" under Known
limitations, which leaves every CFRD result unable to separate the
architecture from its parameter count and training budget. This script runs the
missing control.

Matching parameters alone is not enough. CFRD reuses its physical cells across
recurrences, so it spends far more compute than a dense decoder holding the
same parameters. Two baselines bracket it:

    parameter-matched   equal parameters, fewer FLOPs than CFRD
    compute-matched     equal FLOPs and sequential depth, more parameters

CFRD has to beat the first for folding to be worth anything at all, and the
second for a per-parameter claim. Both baselines see identical windows in an
identical order under an identical schedule, so the architecture is the only
free variable.

A short CPU run sets direction; it does not replace a full training run. Use
--scale release on a GPU to reproduce a release comparison.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

import config
from baseline_model import BaselineConfig
from cfrd_features import SURFACE_FEATURE_DIM
from counterfactual_data import CounterfactualSampler
from counterfactual_objective import counterfactual_ranking_result, encode_counterfactual_pairs
from data_utils import prepare_text_for_tokenizer
from model import ModelConfig, count_parameters
from model_factory import architecture_of_config, build_model, describe_architecture
from surface_features import build_surface_feature_table
from tokenizer_utils import StoryTokenizer
from train import autocast_context, configure_optimizer, get_random_batch, resolve_device

DATA_SEED_OFFSET = 40_000
VALIDATION_SEED_OFFSET = 41_000
RELATION_SEED_OFFSET = 42_000
RELATION_EVAL_SEED_OFFSET = 43_000


def baseline_parameter_count(vocab_size: int, d_model: int, n_head: int, n_kv_head: int, ffn: int, layers: int) -> int:
    """Closed-form parameter count for BaselineLanguageModel.

    Used to size a baseline without building a candidate for every trial.
    Verified against the constructed model in test_matched_sizing_is_exact.
    """

    head_dim = d_model // n_head
    attention = 2 * d_model * d_model + 2 * (d_model * n_kv_head * head_dim)
    per_layer = attention + 3 * d_model * ffn + 2 * d_model
    return vocab_size * d_model + layers * per_layer + d_model + SURFACE_FEATURE_DIM * d_model + 1


def ffn_granularity(d_model: int) -> int:
    """Rounding step for a solved FFN width.

    Widths are kept on a round boundary so the baseline is not handicapped by
    an awkward matmul shape. The step is capped at d_model / 8 so a small test
    configuration is not forced onto a stride coarse enough to break the
    parameter match it exists to verify.
    """

    return max(8, min(32, d_model // 8))


def solve_ffn_dim(target_parameters: int, layers: int, vocab_size: int, d_model: int, n_head: int, n_kv_head: int):
    """Pick the FFN width whose parameter count lands closest to the target."""

    head_dim = d_model // n_head
    attention = 2 * d_model * d_model + 2 * (d_model * n_kv_head * head_dim)
    fixed = vocab_size * d_model + d_model + SURFACE_FEATURE_DIM * d_model + 1 + layers * (attention + 2 * d_model)
    exact = (target_parameters - fixed) / (layers * 3 * d_model)
    if exact <= 0:
        raise ValueError(f"{layers} layers already exceed the parameter budget before any FFN")

    step = ffn_granularity(d_model)
    return max(step, int(round(exact / step) * step))


def match_baselines(cfrd_cfg: ModelConfig, cfrd_parameters: int) -> dict[str, BaselineConfig]:
    """Derive both matched baselines from a CFRD configuration.

    The parameter-matched depth is chosen so ffn_dim lands nearest 3.0x d_model,
    CFRD's own cell proportion. Forcing a specific depth instead would either
    thin the FFN into a handicap or fatten it past anything the architecture
    itself uses, and the difference would then be the FFN, not the folding.
    """

    shared = {
        "vocab_size": cfrd_cfg.vocab_size,
        "context_length": cfrd_cfg.context_length,
        "d_model": cfrd_cfg.d_model,
        "n_head": cfrd_cfg.n_head,
        "n_kv_head": cfrd_cfg.n_kv_head,
        "rope_theta": cfrd_cfg.rope_theta,
        "dropout": cfrd_cfg.dropout,
        "use_surface_features": cfrd_cfg.use_surface_features,
        "surface_feature_gain_init": cfrd_cfg.surface_feature_gain_init,
    }
    sizing = {key: shared[key] for key in ("vocab_size", "d_model", "n_head", "n_kv_head")}

    target_ratio = 3.0
    best_layers, best_ffn, best_distance = None, None, math.inf
    for layers in range(2, cfrd_cfg.recurrences + 4):
        try:
            ffn = solve_ffn_dim(cfrd_parameters, layers, **sizing)
        except ValueError:
            continue
        distance = abs(ffn / cfrd_cfg.d_model - target_ratio)
        if distance < best_distance:
            best_layers, best_ffn, best_distance = layers, ffn, distance

    if best_layers is None:
        raise ValueError("No dense depth fits the CFRD parameter budget")

    # CFRD applies one cell per recurrence, plus the binding block if enabled.
    compute_layers = cfrd_cfg.recurrences + (1 if cfrd_cfg.use_binding_block else 0)

    return {
        "baseline-param-matched": BaselineConfig(n_layer=best_layers, ffn_dim=best_ffn, **shared),
        "baseline-compute-matched": BaselineConfig(n_layer=compute_layers, ffn_dim=best_ffn, **shared),
    }


def cfrd_ablations(cfrd_cfg: ModelConfig) -> dict[str, ModelConfig]:
    """Within-CFRD arms that separate the changes v1.1 shipped together.

    v1.1 added a third physical cell, a full-context binding block, a larger
    vocabulary, and an auxiliary relation objective in one release, then
    reported a combined gain. RESEARCH.md already requires these components to
    be ablated rather than credited jointly. Two of them are structural and can
    be isolated by rebuilding the same configuration with one field changed:

    - cfrd-no-binding-block removes the only full-context path. If the measured
      binding gain survives, it came from folding; if it disappears, v1.1's
      result is a plain attention block wearing a recurrent architecture.
    - cfrd-unfolded gives every recurrence its own cell, holding depth and the
      binding block fixed while removing parameter sharing. It costs more
      parameters by construction, which is the point: it prices the sharing.
    """

    variants: dict[str, ModelConfig] = {}
    if cfrd_cfg.use_binding_block:
        variants["cfrd-no-binding-block"] = replace(cfrd_cfg, use_binding_block=False)
    if cfrd_cfg.physical_cells < cfrd_cfg.recurrences:
        variants["cfrd-unfolded"] = replace(cfrd_cfg, physical_cells=cfrd_cfg.recurrences)
    return variants


def load_token_stream(tokenizer: StoryTokenizer, lines: int, path: Path) -> np.ndarray:
    """Encode the first `lines` records of the corpus into one packed stream."""

    if not path.exists():
        raise FileNotFoundError(f"No corpus at {path}. Place training text under data/ first.")

    ids: list[int] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for index, line in enumerate(handle):
            if index >= lines:
                break
            text = line.strip()
            if not text:
                continue
            ids.append(tokenizer.bos_id)
            ids.extend(tokenizer.sp.encode(prepare_text_for_tokenizer(text), out_type=int))
            ids.append(tokenizer.eos_id)

    return np.asarray(ids, dtype=np.uint16)


def measure_forward_flops(model: torch.nn.Module, context_length: int, vocab_size: int) -> float:
    counter = FlopCounterMode(display=False)
    tokens = torch.randint(0, vocab_size, (1, context_length))
    with counter, torch.no_grad():
        model(tokens)
    return counter.get_total_flops() / 1e9


@torch.no_grad()
def validation_loss(model: torch.nn.Module, data: np.ndarray, device: torch.device, batches: int, batch: int) -> float:
    """Score fixed windows so every architecture is judged on identical text.

    The window seed is deliberately independent of the training seed, so every
    arm of every replication is scored on exactly the same text.
    """

    was_training = model.training
    model.eval()
    rng = np.random.default_rng(config.SEED + VALIDATION_SEED_OFFSET)
    total = 0.0

    for _ in range(batches):
        x, y = get_random_batch(data, batch, model.cfg.context_length, device, rng)
        with autocast_context(device):
            output = model(x, targets=y)
        assert output.final_loss is not None
        total += float(output.final_loss.item())

    if was_training:
        model.train()
    return total / batches


@torch.no_grad()
def relation_metrics(
    model: torch.nn.Module,
    tokenizer: StoryTokenizer,
    device: torch.device,
    pairs: int,
) -> dict[str, float]:
    """Score held-out entity bindings, the capability v1.1 actually claims.

    Validation entities and phrasings are disjoint from the training sampler,
    and the seed is fixed, so every arm faces the same held-out pairs.
    """

    was_training = model.training
    model.eval()
    sampler = CounterfactualSampler("validation")
    rng = np.random.default_rng(config.SEED + RELATION_EVAL_SEED_OFFSET)
    batch = encode_counterfactual_pairs(
        sampler.sample_batch(pairs, rng),
        tokenizer,
        model.cfg.context_length,
        device,
    )

    with autocast_context(device):
        result = counterfactual_ranking_result(model, batch, config.COUNTERFACTUAL_MARGIN)

    if was_training:
        model.train()
    return {
        "relation_loss": float(result.loss.item()),
        "relation_decision_accuracy": float(result.decision_accuracy.item()),
        "relation_strict_pair_accuracy": float(result.strict_pair_accuracy.item()),
        "relation_mean_margin": float(result.mean_margin.item()),
    }


def train_arm(
    name: str,
    model_cfg,
    surface_features: torch.Tensor,
    train_data: np.ndarray,
    val_data: np.ndarray,
    device: torch.device,
    args: argparse.Namespace,
    seed: int,
    tokenizer: StoryTokenizer,
) -> dict:
    """Train one architecture under one seed.

    Within a replication every arm shares the seed, so initialization and the
    window order are identical across architectures and the comparison stays
    paired. Across replications the seed changes, which is what makes a delta
    separable from run-to-run variance.
    """

    torch.manual_seed(seed)
    model = build_model(model_cfg, surface_features)
    parameters = count_parameters(model)["total"]
    gflops = measure_forward_flops(model, model_cfg.context_length, model_cfg.vocab_size)

    model.to(device)
    model.train()
    optimizer = configure_optimizer(model, device)
    rng = np.random.default_rng(seed + DATA_SEED_OFFSET)
    # Relation sampling is kept off the LM window generator, matching train.py,
    # so enabling the objective does not shift which text an arm sees.
    relation_rng = np.random.default_rng(seed + RELATION_SEED_OFFSET)
    relation_sampler = CounterfactualSampler("train") if args.relation_weight > 0.0 else None

    print(f"\n{name} (seed {seed}): {describe_architecture(model_cfg)}", flush=True)
    print(f"  parameters={parameters:,}  forward={gflops:.3f} GFLOPs/{model_cfg.context_length}tok", flush=True)

    history: list[dict] = []
    start = time.perf_counter()

    for step in range(args.steps):
        learning_rate = args.learning_rate * min(1.0, (step + 1) / max(1, args.warmup_steps))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

        optimizer.zero_grad(set_to_none=True)
        x, y = get_random_batch(train_data, args.batch_size, model_cfg.context_length, device, rng)
        with autocast_context(device):
            output = model(x, targets=y)
            assert output.loss is not None
            total_loss = output.loss

            if relation_sampler is not None:
                relation_batch = encode_counterfactual_pairs(
                    relation_sampler.sample_batch(args.relation_pairs, relation_rng),
                    tokenizer,
                    model_cfg.context_length,
                    device,
                )
                relation = counterfactual_ranking_result(model, relation_batch, config.COUNTERFACTUAL_MARGIN)
                total_loss = total_loss + args.relation_weight * relation.loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
        optimizer.step()

        if (step + 1) % args.log_interval == 0 or step == 0:
            assert output.final_loss is not None
            elapsed = time.perf_counter() - start
            print(
                f"  step {step + 1:>4}/{args.steps}  train={float(output.final_loss.item()):.4f}  {elapsed:6.1f}s",
                flush=True,
            )
            history.append({"step": step + 1, "train_loss": float(output.final_loss.item())})

    wall_seconds = time.perf_counter() - start
    final_loss = validation_loss(model, val_data, device, args.eval_batches, args.batch_size)
    relation = relation_metrics(model, tokenizer, device, args.relation_eval_pairs)

    print(f"  validation loss={final_loss:.4f}  perplexity={math.exp(min(final_loss, 20.0)):.3f}", flush=True)
    print(
        f"  held-out relations: strict pairs={relation['relation_strict_pair_accuracy']:.3f}  "
        f"decisions={relation['relation_decision_accuracy']:.3f}  margin={relation['relation_mean_margin']:+.4f}",
        flush=True,
    )

    return {
        "name": name,
        "seed": seed,
        "architecture": architecture_of_config(model_cfg),
        "shape": describe_architecture(model_cfg),
        "model_config": asdict(model_cfg),
        "parameters": parameters,
        "forward_gflops": gflops,
        "validation_loss": final_loss,
        "validation_perplexity": float(math.exp(min(final_loss, 20.0))),
        "wall_seconds": wall_seconds,
        "history": history,
        **relation,
    }


def summarize_arm(name: str, runs: list[dict], reference_runs: list[dict] | None) -> dict:
    """Aggregate one arm's replications and pair its deltas against CFRD.

    Deltas are computed per seed and then averaged, not taken between the two
    averages. Both arms of a replication share initialization and window order,
    so the paired difference cancels the variance the seed introduces.
    """

    losses = [run["validation_loss"] for run in runs]
    mean = sum(losses) / len(losses)
    spread = (sum((value - mean) ** 2 for value in losses) / (len(losses) - 1)) ** 0.5 if len(losses) > 1 else 0.0
    strict = [run["relation_strict_pair_accuracy"] for run in runs]

    summary = {
        "name": name,
        "architecture": runs[0]["architecture"],
        "shape": runs[0]["shape"],
        "parameters": runs[0]["parameters"],
        "forward_gflops": runs[0]["forward_gflops"],
        "seeds": [run["seed"] for run in runs],
        "validation_losses": losses,
        "mean_validation_loss": mean,
        "std_validation_loss": spread,
        "mean_validation_perplexity": float(math.exp(min(mean, 20.0))),
        "strict_pair_accuracies": strict,
        "mean_strict_pair_accuracy": sum(strict) / len(strict),
        "mean_relation_margin": sum(run["relation_mean_margin"] for run in runs) / len(runs),
    }

    if reference_runs is not None:
        by_seed = {run["seed"]: run["validation_loss"] for run in reference_runs}
        deltas = [run["validation_loss"] - by_seed[run["seed"]] for run in runs if run["seed"] in by_seed]
        if deltas:
            delta_mean = sum(deltas) / len(deltas)
            delta_spread = (
                (sum((value - delta_mean) ** 2 for value in deltas) / (len(deltas) - 1)) ** 0.5
                if len(deltas) > 1
                else 0.0
            )
            summary["paired_deltas"] = deltas
            summary["mean_paired_delta"] = delta_mean
            summary["std_paired_delta"] = delta_spread
            # A delta smaller than the spread it sits in is not a result.
            summary["delta_exceeds_spread"] = len(deltas) > 1 and abs(delta_mean) > delta_spread

    return summary


def small_scale_config(vocab_size: int) -> ModelConfig:
    """A CPU-sized CFRD that keeps the release architecture's proportions."""

    return ModelConfig(
        vocab_size=vocab_size,
        context_length=256,
        chunk_size=64,
        d_model=192,
        n_head=6,
        n_kv_head=2,
        ffn_dim=512,
        summary_slots=4,
        memory_dim=64,
        memory_heads=4,
        physical_cells=config.PHYSICAL_CELLS,
        recurrences=4,
        exit_depths=(2, 4),
        use_binding_block=config.USE_BINDING_BLOCK,
        use_surface_features=config.USE_KOREAN_SURFACE_FEATURES,
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--scale",
        choices=("small", "release"),
        default="small",
        help="'small' is a CPU-feasible direction test; 'release' uses config.py's architecture",
    )
    parser.add_argument("--steps", type=int, default=300, help="Optimizer steps per architecture")
    parser.add_argument(
        "--seeds",
        type=int,
        default=3,
        help="Replications per architecture. Deltas between compact models are small enough that a "
        "single seed cannot separate them from run-to-run variance.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--warmup-steps", type=int, default=30)
    parser.add_argument("--corpus-lines", type=int, default=30_000, help="Records read from the corpus")
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument(
        "--relation-weight",
        type=float,
        default=config.COUNTERFACTUAL_LOSS_WEIGHT,
        help="Auxiliary relation-binding loss weight, applied identically to every arm. "
        "Set 0 to compare plain language modelling only. Entity binding is what v1.1 claims, "
        "so leaving this off measures the one thing CFRD was not built for.",
    )
    parser.add_argument(
        "--ablate",
        action="store_true",
        help="Add within-CFRD arms that remove the binding block and parameter sharing, so v1.1's "
        "bundled changes can be credited individually instead of jointly.",
    )
    parser.add_argument("--relation-pairs", type=int, default=config.COUNTERFACTUAL_PAIRS_PER_MICRO_BATCH)
    parser.add_argument("--relation-eval-pairs", type=int, default=config.COUNTERFACTUAL_EVAL_PAIRS)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--device", default=None, help="Defaults to config.DEVICE resolution")
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=None,
        help="SentencePiece model; defaults to config.TOKENIZER_MODEL_PATH. Any vocabulary works, "
        "because all arms share whichever one is given.",
    )
    parser.add_argument("--output", type=Path, default=Path("runs") / "architecture_comparison.json")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    device = torch.device(args.device) if args.device else resolve_device()

    tokenizer = StoryTokenizer(args.tokenizer)
    surface_features = build_surface_feature_table(tokenizer)

    cfrd_cfg = (
        small_scale_config(tokenizer.vocab_size)
        if args.scale == "small"
        else ModelConfig.from_project_settings(config, tokenizer.vocab_size)
    )

    torch.manual_seed(config.SEED)
    cfrd_parameters = count_parameters(build_model(cfrd_cfg, surface_features))["total"]
    # Baselines are sized against the full CFRD, never against an ablated one,
    # so every arm in the table is matched to the same reference.
    architectures = {"cfrd": cfrd_cfg, **match_baselines(cfrd_cfg, cfrd_parameters)}
    if args.ablate:
        architectures.update(cfrd_ablations(cfrd_cfg))

    print(f"Loading up to {args.corpus_lines:,} records from {config.DATA_DIR / 'data.txt'}", flush=True)
    stream = load_token_stream(tokenizer, args.corpus_lines, config.DATA_DIR / "data.txt")
    split = int(len(stream) * 0.98)
    train_data, val_data = stream[:split], stream[split:]
    print(f"train tokens={len(train_data):,}  validation tokens={len(val_data):,}", flush=True)

    seeds = [config.SEED + offset for offset in range(args.seeds)]
    runs: dict[str, list[dict]] = {name: [] for name in architectures}
    # Seed varies in the outer loop so a partial run still holds every arm of
    # each completed replication, which is the unit the pairing needs.
    for seed in seeds:
        for name, model_cfg in architectures.items():
            runs[name].append(
                train_arm(name, model_cfg, surface_features, train_data, val_data, device, args, seed, tokenizer)
            )

    reference_name = next(iter(architectures))
    summaries = [
        summarize_arm(name, arm_runs, None if name == reference_name else runs[reference_name])
        for name, arm_runs in runs.items()
    ]

    tokens_per_arm = args.batch_size * args.steps * cfrd_cfg.context_length
    print("\n" + "=" * 92, flush=True)
    print(f"{args.seeds} seed(s), {args.steps} steps, {tokens_per_arm:,} tokens per arm per seed", flush=True)
    print(
        f"{'arm':<26}{'params':>12}{'GFLOPs':>9}{'val loss':>10}{'sd':>8}{'vs cfrd':>10}{'verdict':>14}{'strict':>9}",
        flush=True,
    )
    print("-" * 92, flush=True)
    for row in summaries:
        if "mean_paired_delta" not in row:
            delta, verdict = "  (ref)", ""
        else:
            delta = f"{row['mean_paired_delta']:+9.4f}"
            if args.seeds < 2:
                verdict = "1 seed"
            elif not row["delta_exceeds_spread"]:
                verdict = "within noise"
            else:
                verdict = "cfrd wins" if row["mean_paired_delta"] > 0 else "cfrd loses"
        print(
            f"{row['name']:<26}{row['parameters']:>12,}{row['forward_gflops']:>9.2f}"
            f"{row['mean_validation_loss']:>10.4f}{row['std_validation_loss']:>8.4f}{delta:>10}{verdict:>14}"
            f"{row['mean_strict_pair_accuracy']:>9.3f}",
            flush=True,
        )
    print("=" * 92, flush=True)
    print("Lower validation loss is better. A positive delta means CFRD won that comparison.", flush=True)
    print(
        f"'strict' is held-out strict pair accuracy at relation weight {args.relation_weight}; "
        "chance is 0.25 and both directions must flip.",
        flush=True,
    )
    if args.seeds < 2:
        print("Run --seeds 3 or more before reading any delta as a result.", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "scale": args.scale,
        "device": str(device),
        "seeds": seeds,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "relation_weight": args.relation_weight,
        "relation_pairs_per_step": args.relation_pairs,
        "relation_eval_pairs": args.relation_eval_pairs,
        "tokenizer": str(tokenizer.model_path),
        "vocab_size": tokenizer.vocab_size,
        "train_tokens": int(len(train_data)),
        "validation_tokens": int(len(val_data)),
        "summary": summaries,
        "results": [run for arm_runs in runs.values() for run in arm_runs],
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
