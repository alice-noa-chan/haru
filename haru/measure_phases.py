"""Measure real optimizer steps for the larger teacher and own-teacher distillation."""

from __future__ import annotations

import gc
import json
import time

import torch

from dense_model import CANDIDATES, DenseLanguageModel, grow_teacher, student_from_teacher
from haru.archive import file_hash
from haru.data import MixtureSampler
from haru.kernels import attention_context
from haru.runtime import atomic_json
from haru.training import distillation_loss, precision_context, source_commit
from surface_features import build_surface_feature_table
from tokenizer_utils import StoryTokenizer


def phase_loss(model, teacher, x, y, loss_backend):
    if teacher is None:
        return model(x, targets=y, loss_only=True, loss_backend=loss_backend).loss
    with torch.no_grad():
        teacher_logits = teacher(x).logits
    return distillation_loss(model(x).logits, teacher_logits, y)


def verify_backend(model, teacher, device, attention_backend, loss_backend):
    ids = torch.randint(4, model.cfg.vocab_size, (1, model.cfg.context_length + 1), device=device)
    x, y = ids[:, :-1], ids[:, 1:].clone()
    y[:, : min(128, model.cfg.context_length // 2)] = -100
    model.zero_grad(set_to_none=True)
    with attention_context("math"), precision_context(device):
        reference = phase_loss(model, teacher, x, y, "torch")
    reference.backward()
    gradients = {name: p.grad.detach().float().clone() for name, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    with attention_context(attention_backend), precision_context(device):
        actual = phase_loss(model, teacher, x, y, loss_backend)
    torch.testing.assert_close(actual.float(), reference.detach().float(), atol=0.01, rtol=0.005)
    actual.backward()
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.grad.float(), gradients[name], atol=0.002, rtol=0.05)
    model.zero_grad(set_to_none=True)


def measure_steps(model, teacher, sampler, device, settings, deadline, effective_tokens=131072, repeats=2):
    batch = settings["microbatch"]
    context = model.cfg.context_length
    if effective_tokens % (batch * context) or repeats < 1:
        raise ValueError("Measurement must use complete optimizer steps and a positive repeat count")
    accumulation = effective_tokens // (batch * context)
    model.train()
    if teacher is not None:
        teacher.eval().requires_grad_(False)
    decay, no_decay = [], []
    for parameter in model.parameters():
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0}],
        lr=3e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=str(device).startswith("cuda"),
    )
    samples = []
    peak = 0
    for index in range(repeats + 1):
        if deadline.expired():
            return None
        model.zero_grad(set_to_none=True)
        if str(device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        started = time.monotonic()
        for _ in range(accumulation):
            if deadline.expired():
                return None
            x, y = sampler.batch(batch, context)
            x, y = torch.as_tensor(x, device=device), torch.as_tensor(y, device=device)
            with attention_context(settings["attention_backend"]), precision_context(device):
                loss = phase_loss(model, teacher, x, y, settings["loss_backend"])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite measurement loss")
            (loss / accumulation).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
            peak = max(peak, torch.cuda.max_memory_allocated())
        if index:
            samples.append(time.monotonic() - started)
    model.zero_grad(set_to_none=True)
    return {
        **settings,
        "effective_tokens": effective_tokens,
        "timed_optimizer_steps": repeats,
        "tokens_per_second": repeats * effective_tokens / sum(samples),
        "step_seconds": samples,
        "peak_bytes": peak,
        "output_and_gradient_agreement_verified": True,
        "includes_sampler_optimizer_and_teacher": True,
    }


def measure_phases(data, output, configuration, deadline):
    fingerprint = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "data_manifest_blake2b16": file_hash(data / "manifest.json"),
        "configuration": configuration,
    }
    previous = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {}
    if previous.get("fingerprint") == fingerprint and previous.get("status") == "complete":
        return previous
    if previous:
        atomic_json(output.with_name(f"phase_throughput_history_{time.time_ns()}.json"), previous)
    tokenizer = StoryTokenizer(data / "tokenizer.model")
    features = build_surface_feature_table(tokenizer)
    result = {
        "fingerprint": fingerprint,
        "source_commit": source_commit(),
        "status": "running",
        "measurements": [],
        "failures": [],
    }
    for candidate, cfg in CANDIDATES.items():
        for phase in ("teacher-base", "student-base"):
            if deadline.expired():
                atomic_json(output, result)
                return result
            base = DenseLanguageModel(cfg, features).cuda()
            teacher = grow_teacher(base)
            del base
            if phase == "teacher-base":
                model, frozen_teacher = teacher, None
                loss_backend = configuration[candidate].get("loss_backend", "torch")
            else:
                model, frozen_teacher = student_from_teacher(teacher), teacher.eval().requires_grad_(False)
                loss_backend = "torch"
            backend = configuration[candidate].get("attention_backend", "auto")
            verify_backend(model, frozen_teacher, "cuda", backend, loss_backend)
            for batch in (4, 8, 16, 32, 64, 128):
                try:
                    settings = {
                        "microbatch": batch,
                        "attention_backend": backend,
                        "loss_backend": loss_backend,
                        "compiled": False,
                    }
                    row = measure_steps(model, frozen_teacher, MixtureSampler(data, 1337), "cuda", settings, deadline)
                    if row is None:
                        break
                    row.update({"candidate": candidate, "phase": phase})
                    result["measurements"].append(row)
                    print(json.dumps(row), flush=True)
                    atomic_json(output, result)
                except torch.OutOfMemoryError:
                    result["failures"].append(
                        {"candidate": candidate, "phase": phase, "microbatch": batch, "error": "out_of_memory"}
                    )
                    break
                finally:
                    model.zero_grad(set_to_none=True)
                    gc.collect()
                    torch.cuda.empty_cache()
            del model, teacher, frozen_teacher
            gc.collect()
            torch.cuda.empty_cache()
    result["status"] = (
        "complete"
        if not deadline.expired()
        and all(
            any(row["candidate"] == candidate and row["phase"] == phase for row in result["measurements"])
            for candidate in CANDIDATES
            for phase in ("teacher-base", "student-base")
        )
        else "running"
    )
    atomic_json(output, result)
    return result
