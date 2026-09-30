"""Optional, measured execution backends; checkpoints stay CPU-portable."""

from __future__ import annotations

import gc
import time
from contextlib import nullcontext

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel


def attention_context(backend="auto"):
    if backend == "auto":
        return nullcontext()
    return sdpa_kernel(
        {"flash": SDPBackend.FLASH_ATTENTION, "cudnn": SDPBackend.CUDNN_ATTENTION, "math": SDPBackend.MATH}[backend]
    )


def attention_trace(model, ids):
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as trace:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model(ids)
    return sorted({event.key for event in trace.key_averages() if "scaled_dot_product" in event.key})


def benchmark(data, output, deadline):
    from dense_model import CANDIDATES, DenseLanguageModel
    from haru.runtime import Deadline, atomic_json
    from surface_features import build_surface_feature_table
    from tokenizer_utils import StoryTokenizer

    tokenizer = StoryTokenizer(data / "tokenizer.model")
    features = build_surface_feature_table(tokenizer)
    results, failures, traces = [], [], {}
    compile_deadline = Deadline(seconds=300)

    def record():
        atomic_json(
            output,
            {
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "status": "complete" if complete else "running",
                "measurements": results,
                "failed_configurations": failures,
                "attention_operators": traces,
            },
        )

    complete = False
    for name, cfg in CANDIDATES.items():
        if deadline.expired():
            break
        torch.manual_seed(1337)
        model = DenseLanguageModel(cfg, features).cuda().train()
        ids = torch.randint(4, cfg.vocab_size, (2, cfg.context_length + 1), device="cuda")
        x, y = ids[:, :-1], ids[:, 1:].clone()
        y[0, :128] = -100
        with torch.no_grad():
            fp32 = model(x).logits.float()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            reference = model(x).logits.float()
        torch.testing.assert_close(reference, fp32, atol=0.04, rtol=0.04)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            reference_loss = model(x, targets=y, loss_only=True).loss
        reference_loss.backward()
        gradients = {key: parameter.grad.detach().float().clone() for key, parameter in model.named_parameters()}
        traces[name] = attention_trace(model, x)

        def trial(
            backend, loss_backend, compiled, batches, model=model, reference_loss=reference_loss, gradients=gradients
        ):
            if deadline.expired() or (compiled and compile_deadline.expired()):
                return
            try:
                run_model = torch.compile(model, dynamic=True) if compiled else model
                model.zero_grad(set_to_none=True)
                with attention_context(backend), torch.autocast("cuda", dtype=torch.bfloat16):
                    actual = run_model(x, targets=y, loss_only=True, loss_backend=loss_backend).loss
                torch.testing.assert_close(actual.float(), reference_loss.detach().float(), atol=0.01, rtol=0.005)
                actual.backward()
                for key, parameter in model.named_parameters():
                    torch.testing.assert_close(parameter.grad.float(), gradients[key], atol=0.002, rtol=0.05)
                for batch in batches:
                    if deadline.expired() or (compiled and compile_deadline.expired()):
                        break
                    try:
                        model.zero_grad(set_to_none=True)
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()
                        samples = torch.randint(4, cfg.vocab_size, (batch, cfg.context_length + 1), device="cuda")
                        for _ in range(2):
                            model.zero_grad(set_to_none=True)
                            with attention_context(backend), torch.autocast("cuda", dtype=torch.bfloat16):
                                loss = run_model(
                                    samples[:, :-1], targets=samples[:, 1:], loss_only=True, loss_backend=loss_backend
                                ).loss
                            loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
                        torch.cuda.synchronize()
                        started = time.monotonic()
                        for _ in range(3):
                            model.zero_grad(set_to_none=True)
                            with attention_context(backend), torch.autocast("cuda", dtype=torch.bfloat16):
                                loss = run_model(
                                    samples[:, :-1], targets=samples[:, 1:], loss_only=True, loss_backend=loss_backend
                                ).loss
                            loss.backward()
                        torch.cuda.synchronize()
                        row = {
                            "candidate": name,
                            "attention_backend": backend,
                            "loss_backend": loss_backend,
                            "compiled": compiled,
                            "microbatch": batch,
                            "tokens_per_second": 3 * batch * cfg.context_length / (time.monotonic() - started),
                            "peak_bytes": torch.cuda.max_memory_allocated(),
                            "precision": "bf16",
                            "output_and_gradient_agreement_verified": True,
                        }
                        results.append(row)
                        print(row, flush=True)
                        record()
                    except torch.OutOfMemoryError:
                        model.zero_grad(set_to_none=True)
                        torch.cuda.empty_cache()
                        failures.append(
                            {
                                "candidate": name,
                                "attention": backend,
                                "loss": loss_backend,
                                "compiled": compiled,
                                "microbatch": batch,
                                "error": "out_of_memory",
                            }
                        )
                        break
            except Exception as error:
                failures.append(
                    {
                        "candidate": name,
                        "attention": backend,
                        "loss": loss_backend,
                        "compiled": compiled,
                        "error": str(error)[:700],
                    }
                )
                model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                record()
                if backend == "auto" and loss_backend == "torch" and not compiled:
                    raise

        trial("auto", "torch", False, (4, 8, 16, 32, 64, 128))
        baseline = max(
            (r for r in results if r["candidate"] == name), key=lambda r: r["tokens_per_second"], default=None
        )
        if baseline:
            for backend in ("flash", "cudnn"):
                trial(backend, "torch", False, (baseline["microbatch"],))
            best = max((r for r in results if r["candidate"] == name), key=lambda r: r["tokens_per_second"])
            trial(best["attention_backend"], "liger", False, (16, 32, 64, 128))
            best = max((r for r in results if r["candidate"] == name), key=lambda r: r["tokens_per_second"])
            trial(best["attention_backend"], best["loss_backend"], True, (best["microbatch"],))
        del trial, model, gradients, fp32, reference, reference_loss
        gc.collect()
        torch.cuda.empty_cache()
    complete = all(any(row["candidate"] == name for row in results) for name in CANDIDATES)
    record()
    return results
