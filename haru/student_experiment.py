"""Compare the paired-FFN students with one frozen teacher, then train separate IT checkpoints."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from dense_model import EXPERIMENTAL_CANDIDATES, student_from_teacher
from haru.archive import file_hash
from haru.data import MixtureSampler
from haru.kernels import attention_trace
from haru.measure_phases import measure_steps, verify_backend
from haru.pipeline import select
from haru.runtime import Deadline, atomic_json
from haru.training import checkpoint_validation, load_model, source_commit, train
from haru.training import parser as training_parser


def measure_student(teacher_path, candidate, data, output, deadline, device):
    """Benchmark complete KD optimizer steps; these temporary weights are discarded."""
    teacher, _ = load_model(teacher_path, device)
    teacher.eval().requires_grad_(False)
    cfg = EXPERIMENTAL_CANDIDATES[candidate]
    model = student_from_teacher(teacher, cfg)
    verify_backend(model, teacher, device, "auto", "torch")
    rows, failures = [], []
    trace = attention_trace(model, torch.randint(4, cfg.vocab_size, (1, 63), device=device))
    del model
    for batch in (8, 16, 32):
        if deadline.expired():
            break
        sampler = MixtureSampler(data, 1337)
        model = student_from_teacher(teacher, cfg)
        try:
            settings = {"microbatch": batch, "attention_backend": "auto", "loss_backend": "torch", "compiled": False}
            row = measure_steps(model, teacher, sampler, device, settings, deadline)
            if row is None:
                break
            rows.append(row)
        except torch.OutOfMemoryError:
            failures.append({"microbatch": batch, "error": "out_of_memory"})
            break
        finally:
            sampler.close()
            del model
            gc.collect()
            torch.cuda.empty_cache()
    result = {
        "candidate": candidate,
        "measurements": rows,
        "failures": failures,
        "attention_operators": trace,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "source_commit": source_commit(),
        "teacher_blake2b16": file_hash(teacher_path),
        "data_manifest_blake2b16": file_hash(data / "manifest.json"),
    }
    if rows:
        result["selected"] = max(rows, key=lambda row: row["tokens_per_second"])
    atomic_json(output, result)
    del teacher
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_experiment(args, deadline=None):
    deadline = deadline or Deadline(args.deadline_unix, args.max_seconds)
    args.output.mkdir(parents=True, exist_ok=True)
    teacher_hash = file_hash(args.teacher)
    rows = []
    for candidate in EXPERIMENTAL_CANDIDATES:
        if deadline.expired():
            return {"status": "budget_stop", "phase": candidate}
        settings = {"microbatch": args.microbatch, "attention_backend": "auto", "loss_backend": "torch"}
        if args.device.startswith("cuda"):
            # Remeasure per session/GPU; a saved throughput result from another GPU is not reused.
            measured = measure_student(
                args.teacher, candidate, args.data, args.output / candidate / "throughput.json", deadline, args.device
            )
            if "selected" not in measured:
                return {"status": "budget_stop", "phase": f"measure/{candidate}"}
            settings = measured["selected"]
        options = training_parser().parse_args(
            [
                "--data",
                str(args.data),
                "--teacher",
                str(args.teacher),
                "--output",
                str(args.output / candidate),
                "--candidate",
                candidate,
                "--phase",
                "student-base",
                "--device",
                args.device,
                "--seed",
                str(args.seed),
                "--learning-rate",
                "0.0003",
                "--target-tokens",
                str(args.student_tokens),
                "--schedule-tokens",
                str(args.student_tokens),
                "--effective-tokens",
                "131072",
                "--microbatch",
                str(settings["microbatch"]),
                "--attention-backend",
                settings["attention_backend"],
                "--loss-backend",
                "torch",
            ]
        )
        status = train(options, deadline)
        if file_hash(args.teacher) != teacher_hash:
            raise RuntimeError("Frozen teacher checkpoint changed")
        if status["status"] != "complete":
            return {"status": "budget_stop", "phase": candidate, "checkpoint": status.get("latest")}
        checkpoint = args.output / candidate / "best.pt"
        rows.append(
            {"candidate": candidate, "checkpoint": str(checkpoint), "validation": checkpoint_validation(checkpoint)}
        )
    winner, ranking = select(rows)
    selection = {
        "winner": winner,
        "candidates": ranking,
        "seed": args.seed,
        "student_tokens": args.student_tokens,
        "teacher_blake2b16": teacher_hash,
        "source_commit": source_commit(),
        "limitation": "One training seed; paired document bootstrap does not measure seed variance.",
    }
    atomic_json(args.output / "selection.json", selection)
    if args.stop_after == "student-base":
        return {"status": "phase_complete", "phase": "student-base", "winner": winner["candidate"]}
    candidate = winner["candidate"]
    phases = [
        ("teacher-chat", ["--initialize", str(args.teacher), "--allow-large", "--chat"]),
        (
            "student-chat",
            ["--initialize", winner["checkpoint"], "--teacher", str(args.output / "teacher-chat/best.pt"), "--chat"],
        ),
    ]
    for phase, extra in phases:
        if deadline.expired():
            return {"status": "budget_stop", "phase": phase}
        # Conservative initial microbatch; verify actual loss/gradient backend before each IT phase.
        initialized, _ = load_model(args.teacher if phase == "teacher-chat" else winner["checkpoint"], args.device)
        frozen = None
        if phase == "student-chat":
            frozen, _ = load_model(args.output / "teacher-chat/best.pt", args.device)
            frozen.eval().requires_grad_(False)
        verify_backend(initialized, frozen, args.device, "auto", "torch")
        del initialized, frozen
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        options = training_parser().parse_args(
            [
                "--data",
                str(args.data),
                "--output",
                str(args.output / phase),
                "--candidate",
                candidate,
                "--phase",
                phase,
                "--device",
                args.device,
                "--seed",
                str(args.seed),
                "--learning-rate",
                "0.0001",
                "--target-tokens",
                str(args.chat_tokens),
                "--schedule-tokens",
                str(args.chat_tokens),
                "--effective-tokens",
                "131072",
                "--microbatch",
                "8",
                *extra,
            ]
        )
        status = train(options, deadline)
        if file_hash(args.teacher) != teacher_hash:
            raise RuntimeError("Base teacher checkpoint changed during IT training")
        if status["status"] != "complete":
            return {"status": "budget_stop", "phase": phase, "checkpoint": status.get("latest")}
    return {"status": "complete", "winner": candidate, "teacher_blake2b16": teacher_hash}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--student-tokens", type=int, default=500_000_000)
    parser.add_argument("--chat-tokens", type=int, default=10_000_000)
    parser.add_argument("--microbatch", type=int, default=16)
    parser.add_argument("--deadline-unix", type=float)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--stop-after", choices=("student-base", "student-chat"), default="student-chat")
    args = parser.parse_args()
    status = run_experiment(args)
    atomic_json(args.output / "pipeline_status.json", status)
    print(json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
