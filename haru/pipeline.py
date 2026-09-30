"""Budget-interruptible staged comparison, own teacher and four model artifacts."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from dense_model import CANDIDATES
from haru.expand_rules import materialize
from haru.export import export
from haru.kernels import benchmark
from haru.measure_phases import measure_phases
from haru.runtime import Deadline, atomic_json
from haru.training import checkpoint_validation, load_model, train
from haru.training import parser as training_parser


@torch.inference_mode()
def cpu_speed(checkpoint):
    model, _ = load_model(checkpoint)
    model.eval()
    threads = torch.get_num_threads()
    torch.set_num_threads(4)
    ids = torch.arange(256)[None, :] % model.cfg.vocab_size
    samples = []
    try:
        for _ in range(4):
            start = time.perf_counter()
            output = model(ids, use_cache=True, logits_to_keep=1)
            prefill = time.perf_counter() - start
            start = time.perf_counter()
            for _ in range(32):
                output = model(
                    output.logits[:, -1:].argmax(-1),
                    use_cache=True,
                    past_key_values=output.past_key_values,
                    logits_to_keep=1,
                )
            samples.append({"prefill_seconds": prefill, "tokens_per_second": 32 / (time.perf_counter() - start)})
    finally:
        torch.set_num_threads(threads)
    return {
        "prefill_seconds": float(np.median([x["prefill_seconds"] for x in samples[1:]])),
        "tokens_per_second": float(np.median([x["tokens_per_second"] for x in samples[1:]])),
        "threads": 4,
    }


def paired_interval(left, right, trials=2000):
    if left.get("selection_metric", "macro_bpc") != right.get("selection_metric", "macro_bpc"):
        raise ValueError("Candidate validation objectives differ")
    domains = ("story",) if left.get("selection_metric") == "story_bpc" else tuple(left["domains"])
    rng = np.random.default_rng(9182)
    samples = []
    for _ in range(trials):
        changes = []
        for domain in domains:
            a, b = left["domains"][domain], right["domains"][domain]
            scores = np.asarray(a["document_bpc"]) - np.asarray(b["document_bpc"])
            weights = np.asarray(a["document_characters"])
            indices = rng.integers(0, len(scores), len(scores))
            changes.append(float(np.average(scores[indices], weights=weights[indices])))
        samples.append(np.mean(changes))
    return np.quantile(samples, [0.025, 0.975]).tolist()


def select(rows):
    rows = sorted(rows, key=lambda row: row["validation"].get("selection_score", row["validation"]["macro_bpc"]))
    winner = rows[0]
    for alternative in rows[1:]:
        low, high = paired_interval(winner["validation"], alternative["validation"])
        if low <= 0 <= high:
            if "cpu" not in winner:
                winner["cpu"] = cpu_speed(winner["checkpoint"])
            if "cpu" not in alternative:
                alternative["cpu"] = cpu_speed(alternative["checkpoint"])
            if alternative["cpu"]["tokens_per_second"] > winner["cpu"]["tokens_per_second"]:
                winner = alternative
    return winner, rows


def run_pipeline(args):
    args.output.mkdir(parents=True, exist_ok=True)
    deadline = Deadline(args.deadline_unix, args.max_seconds)
    measurement_path = args.output / "throughput.json"
    if args.device == "cuda":
        previous = json.loads(measurement_path.read_text()) if measurement_path.exists() else {}
        if (
            previous.get("gpu") != torch.cuda.get_device_name()
            or previous.get("torch") != torch.__version__
            or previous.get("status") != "complete"
        ):
            if previous:
                atomic_json(args.output / f"throughput_history_{time.time_ns()}.json", previous)
            benchmark(args.data, measurement_path, deadline)
    measurements = json.loads(measurement_path.read_text())["measurements"] if measurement_path.exists() else []
    configuration = {}
    for name in CANDIDATES:
        rows = [x for x in measurements if x["candidate"] == name]
        configuration[name] = (
            max(rows, key=lambda x: x["tokens_per_second"]) if rows else {"microbatch": 8, "compiled": False}
        )
    phase_measurements = []
    if args.device == "cuda" and not deadline.expired():
        report = measure_phases(args.data, args.output / "throughput_phases.json", configuration, deadline)
        if report["status"] != "complete":
            return {"status": "budget_stop", "phase": "throughput_phases"}
        phase_measurements = report["measurements"]

    def phase(name, candidate, lr, target, schedule, seed=1337, extra=()):
        settings = configuration[candidate]
        microbatch = settings["microbatch"] if name.startswith("candidate") else 8
        if name in ("teacher-base", "student-base"):
            rows = [row for row in phase_measurements if row["candidate"] == candidate and row["phase"] == name]
            if rows:
                settings = max(rows, key=lambda row: row["tokens_per_second"])
                microbatch = settings["microbatch"]
        options = [
            "--data",
            str(args.data if name.startswith("candidate") else teacher_data),
            "--output",
            str(args.output / name),
            "--phase",
            name.split("/")[0],
            "--candidate",
            candidate,
            "--learning-rate",
            str(lr),
            "--target-tokens",
            str(target),
            "--schedule-tokens",
            str(schedule),
            "--seed",
            str(seed),
            "--device",
            args.device,
            "--microbatch",
            str(microbatch),
            "--attention-backend",
            settings.get("attention_backend", "auto"),
            "--loss-backend",
            settings.get("loss_backend", "torch"),
        ]
        if configuration[candidate]["compiled"] and name.startswith("candidate"):
            options.append("--compile")
        result = train(training_parser().parse_args([*options, *extra]), deadline)
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()
        return result

    candidates = []

    def selected_validation(name):
        root = args.output / name
        metrics = checkpoint_validation(root / "best.pt")
        if metrics is None:
            raise ValueError(f"Missing selected validation for {name}")
        return metrics

    for candidate in CANDIDATES:
        choice_path = args.output / f"{candidate}.learning_rate.json"
        rates = []
        for rate in () if choice_path.exists() else (3e-4, 1e-3):
            name = f"candidate/{candidate}-{rate}-1337"
            result = phase(name, candidate, rate, 20_000_000, 100_000_000)
            if result["status"] != "complete":
                return {"status": "budget_stop", "phase": name, "checkpoint": result.get("latest")}
            rates.append(
                {
                    "candidate": candidate,
                    "learning_rate": rate,
                    "validation": selected_validation(name),
                    "checkpoint": str(args.output / name / "best.pt"),
                    "name": name,
                }
            )
        if choice_path.exists():
            chosen = json.loads(choice_path.read_text(encoding="utf-8"))
        else:
            chosen, _ = select(rates)
            atomic_json(choice_path, chosen)
        # Resolve saved checkpoints under the current output directory on resume.
        chosen["checkpoint"] = str(args.output / chosen["name"] / "best.pt")
        result = phase(chosen["name"], candidate, chosen["learning_rate"], 100_000_000, 100_000_000)
        if result["status"] != "complete":
            return {"status": "budget_stop", "phase": chosen["name"], "checkpoint": result.get("latest")}
        chosen["validation"] = selected_validation(chosen["name"])
        chosen.pop("cpu", None)
        candidates.append(chosen)
    _, ranking = select(candidates)
    for row in ranking[:2]:
        name = f"candidate/{row['candidate']}-{row['learning_rate']}-1338"
        result = phase(name, row["candidate"], row["learning_rate"], 100_000_000, 100_000_000, seed=1338)
        if result["status"] != "complete":
            return {"status": "budget_stop", "phase": name, "checkpoint": result.get("latest")}
        second_validation = selected_validation(name)
        # Average the two seeds; keep the documented first-seed source for deterministic growth.
        for domain in row["validation"]["domains"]:
            first = row["validation"]["domains"][domain]
            second = second_validation["domains"][domain]
            first["document_bpc"] = ((np.asarray(first["document_bpc"]) + second["document_bpc"]) / 2).tolist()
            first["bpc"] = (first["bpc"] + second["bpc"]) / 2
        row["validation"]["macro_bpc"] = (row["validation"]["macro_bpc"] + second_validation["macro_bpc"]) / 2
        row["validation"]["selection_score"] = (
            row["validation"]["selection_score"] + second_validation["selection_score"]
        ) / 2
    winner, _ = select(ranking[:2])
    atomic_json(args.output / "selection.json", {"winner": winner, "candidates": candidates, "replicated": ranking[:2]})
    candidate = winner["candidate"]
    overlay = args.teacher_rules_overlay
    if overlay is None or not (overlay / "rule_generation.json").exists():
        return {"status": "teacher_data_required", "phase": "teacher-base", "checkpoint": winner["checkpoint"]}
    teacher_data = materialize(args.data, overlay, args.data.parent / "haru-v3-teacher")
    phases = [
        (
            "teacher-base",
            args.teacher_tokens,
            3e-4,
            ["--initialize", winner["checkpoint"], "--grow-teacher", "--allow-large"],
        ),
        ("student-base", args.student_tokens, 3e-4, ["--teacher", str(args.output / "teacher-base/best.pt")]),
        (
            "teacher-chat",
            args.chat_tokens,
            1e-4,
            ["--initialize", str(args.output / "teacher-base/best.pt"), "--allow-large", "--chat"],
        ),
        (
            "student-chat",
            args.chat_tokens,
            1e-4,
            [
                "--initialize",
                str(args.output / "student-base/best.pt"),
                "--teacher",
                str(args.output / "teacher-chat/best.pt"),
                "--chat",
            ],
        ),
    ]
    for name, tokens, lr, extra in phases:
        result = phase(name, candidate, lr, tokens, tokens, extra=extra)
        if result["status"] != "complete":
            return {"status": "budget_stop", "phase": name, "checkpoint": result.get("latest")}
        artifact = args.output / "exports" / name
        if not (artifact / "export_metadata.json").exists():
            export(
                args.output / name / "best.pt",
                teacher_data / "tokenizer.model",
                artifact,
                None,
            )
    return {"status": "complete", "winner": candidate, "exports": str(args.output / "exports")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("packed/haru-v3-story"))
    parser.add_argument("--teacher-rules-overlay", type=Path)
    parser.add_argument("--output", type=Path, default=Path("runs/haru-v3"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--deadline-unix", type=float)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--teacher-tokens", type=int, default=2_000_000_000)
    parser.add_argument("--student-tokens", type=int, default=500_000_000)
    parser.add_argument("--chat-tokens", type=int, default=10_000_000)
    args = parser.parse_args()
    result = run_pipeline(args)
    atomic_json(args.output / "pipeline_status.json", result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
