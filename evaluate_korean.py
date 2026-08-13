"""Score a Haru checkpoint on Korean zero-shot benchmarks.

Until now the only numbers this project produced were its own validation loss
and its own counterfactual pairs, neither of which can be compared against any
other model. KoBEST is the Korean analogue of the English suite small decoders
are usually reported on (ARC, HellaSwag, PIQA, WinoGrande and friends), and
running the same harness everyone else uses is what makes a number comparable.

Haru cannot be evaluated on the English suite at all. Its tokenizer spends 1.02
tokens per English character against 0.29 per Korean character, and only 292 of
its 12,000 pieces are ASCII, 256 of those byte fallbacks. An English benchmark
would measure the tokenizer, not the model.

Scoring goes through lm-eval rather than a hand-written loop, so results are
produced by the same code path as published baselines. The model is loaded
directly from this repository instead of through `trust_remote_code`, because
the exported package spans several files and the dynamic module loader does not
reliably carry their relative imports.

Macro F1 is requested by three of the tasks but the metric script does not
resolve in this environment, so only accuracy is reported. Accuracy is the
figure these benchmarks are usually compared on.

Chance levels matter when reading the output. KoBEST BoolQ, COPA, SentiNeg and
WiC are two-way (50%), HellaSwag is four-way (25%). A compact model near those
numbers has learned nothing about the task.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

KOBEST_TASKS = ("kobest_boolq", "kobest_copa", "kobest_hellaswag", "kobest_sentineg", "kobest_wic")
CHANCE_LEVEL = {
    "kobest_boolq": 0.50,
    "kobest_copa": 0.50,
    "kobest_hellaswag": 0.25,
    "kobest_sentineg": 0.50,
    "kobest_wic": 0.50,
}


def build_harness_model(export_dir: Path, device: str, batch_size: int):
    """Wrap an exported Transformers directory for lm-eval.

    The classes are imported from this repository and registered by hand. Going
    through trust_remote_code instead makes the harness depend on the dynamic
    module loader resolving relative imports across the exported files, which it
    does not do reliably.
    """

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from configuration_cfrd import CFRDConfig
    from modeling_cfrd import CFRDForCausalLM
    from tokenization_cfrd import CFRDTokenizer

    AutoConfig.register("cfrd", CFRDConfig, exist_ok=True)
    AutoModelForCausalLM.register(CFRDConfig, CFRDForCausalLM, exist_ok=True)
    AutoTokenizer.register(CFRDConfig, slow_tokenizer_class=CFRDTokenizer, exist_ok=True)

    hf_config = CFRDConfig.from_pretrained(export_dir)
    model = CFRDForCausalLM.from_pretrained(export_dir, config=hf_config).eval()
    tokenizer = CFRDTokenizer.from_pretrained(export_dir)

    from lm_eval.models.huggingface import HFLM

    return HFLM(pretrained=model, tokenizer=tokenizer, device=device, batch_size=batch_size)


def summarize(results: dict) -> tuple[list[dict], float, float]:
    """Collect per-task accuracy and the mean, alongside each chance level."""

    rows: list[dict] = []
    for task, metrics in sorted(results.items()):
        accuracy = metrics.get("acc,none")
        if accuracy is None:
            continue
        chance = CHANCE_LEVEL.get(task)
        rows.append(
            {
                "task": task,
                "accuracy": float(accuracy),
                "chance": chance,
                "above_chance": None if chance is None else float(accuracy) - chance,
                "macro_f1": metrics.get("f1,none"),
            }
        )

    mean = sum(row["accuracy"] for row in rows) / len(rows) if rows else 0.0
    # The tasks have different chance levels, so a bare mean has no fixed
    # floor. Reporting the mean of the chance levels alongside it is what
    # makes the headline number readable.
    chance_mean = sum(row["chance"] or 0.0 for row in rows) / len(rows) if rows else 0.0
    return rows, mean, chance_mean


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "export_dir",
        type=Path,
        help="A Transformers export directory produced by export_transformers.py",
    )
    parser.add_argument("--tasks", nargs="+", default=list(KOBEST_TASKS))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Score only N examples per task, for checks")
    parser.add_argument("--output", type=Path, default=Path("results") / "korean_benchmarks.json")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if not (args.export_dir / "config.json").exists():
        raise FileNotFoundError(f"No Transformers export at {args.export_dir}. Run export_transformers.py first.")

    import lm_eval

    harness_model = build_harness_model(args.export_dir, args.device, args.batch_size)
    output = lm_eval.simple_evaluate(model=harness_model, tasks=list(args.tasks), limit=args.limit)

    rows, mean, chance_mean = summarize(output["results"])

    print("\n" + "=" * 62, flush=True)
    print(f"{'task':<22}{'accuracy':>10}{'chance':>9}{'vs chance':>12}", flush=True)
    print("-" * 62, flush=True)
    for row in rows:
        delta = "" if row["above_chance"] is None else f"{row['above_chance']:+.3f}"
        chance = "" if row["chance"] is None else f"{row['chance']:.2f}"
        print(f"{row['task']:<22}{row['accuracy']:>10.3f}{chance:>9}{delta:>12}", flush=True)
    print("-" * 62, flush=True)
    print(f"{'mean accuracy':<22}{mean:>10.3f}{chance_mean:>9.3f}{mean - chance_mean:>+12.3f}", flush=True)
    print("=" * 62, flush=True)
    verdict = "above chance" if mean > chance_mean else "at or below chance"
    print(f"Mean is {verdict}. The tasks have different chance levels, so the mean", flush=True)
    print("means nothing without the chance column beside it.", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "export_dir": str(args.export_dir),
        "tasks": list(args.tasks),
        "limit": args.limit,
        "mean_accuracy": mean,
        "chance_mean_accuracy": chance_mean,
        "results": rows,
        "raw": {task: metrics for task, metrics in output["results"].items()},
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
