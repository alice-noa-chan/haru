"""Resumable v3 CE, own-teacher distillation and assistant-only SFT training."""

from __future__ import annotations

import argparse
import json
import math
import random
import signal
import subprocess
import threading
import time
from contextlib import nullcontext
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from data_utils import blake2b_file
from dense_model import CANDIDATES, DenseConfig, DenseLanguageModel, grow_teacher, parameter_count, student_from_teacher
from haru.data import CHAT_TEMPLATE, WEIGHTS, MixtureSampler, data_profile
from haru.kernels import attention_context
from haru.runtime import Deadline, atomic_json
from surface_features import build_surface_feature_table
from tokenizer_utils import StoryTokenizer


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def precision_context(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if str(device).startswith("cuda") else nullcontext()


def distillation_loss(student_logits, teacher_logits, targets, temperature=2.0):
    valid = targets.reshape(-1) != -100
    if not bool(valid.any()):
        raise ValueError("No supervised tokens")
    student = student_logits.reshape(-1, student_logits.size(-1))[valid].float()
    teacher = teacher_logits.reshape(-1, teacher_logits.size(-1))[valid].float()
    gold = targets.reshape(-1)[valid]
    ce = F.cross_entropy(student, gold)
    kd = (
        F.kl_div(
            F.log_softmax(student / temperature, -1),
            F.softmax(teacher.detach() / temperature, -1),
            reduction="batchmean",
        )
        * temperature**2
    )
    return 0.5 * ce + 0.5 * kd


def source_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        provenance = Path(__file__).resolve().parent.parent / "source_commit.txt"
        return provenance.read_text().strip() if provenance.exists() else "unknown"


def save_checkpoint(path, model, optimizer, sampler, step, tokens, best_bpc, training_config, metadata):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_arch": "haru-dense",
        "model_config": asdict(model.cfg),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "sampler": sampler.state_dict(),
        "step": step,
        "tokens_seen": tokens,
        "best_bpc": best_bpc,
        "training_config": training_config,
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        **metadata,
    }
    with temporary.open("wb") as handle:
        import os

        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    atomic_json(
        path.with_suffix(".manifest.json"),
        {
            "file": path.name,
            "bytes": path.stat().st_size,
            "blake2b16": blake2b_file(path),
            "tokens_seen": tokens,
            "step": step,
            **metadata,
        },
    )


def load_model(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_arch") != "haru-dense":
        raise ValueError("Expected a Haru v3 checkpoint")
    cfg = DenseConfig(**checkpoint["model_config"])
    model = DenseLanguageModel(cfg, checkpoint["model"].get("surface_feature_table"))
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device), checkpoint


def checkpoint_validation(path, checkpoint=None):
    """Read metrics belonging to the committed weights, never a newer sidecar."""
    path = Path(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False) if checkpoint is None else checkpoint
    metrics = checkpoint.get("selection_validation")
    if metrics is None:
        record = path.parent / ("validation_best.json" if path.name == "best.pt" else "validation.json")
        metrics = json.loads(record.read_text(encoding="utf-8")) if record.exists() else None
    if metrics is not None and (
        metrics.get("step") != checkpoint["step"] or metrics.get("tokens_seen") != checkpoint["tokens_seen"]
    ):
        raise ValueError("Validation metrics do not belong to this checkpoint")
    return metrics


class ChatSampler:
    def __init__(self, root, tokenizer, seed, split="train"):
        self.rng = np.random.default_rng(seed)
        self.items = []
        self.groups = {"grounded_extract": [], "rules": [], "json_format": [], "story": []}
        with (Path(root) / f"chat.{split}.jsonl").open(encoding="utf-8") as reader:
            for line in reader:
                item = json.loads(line)
                ids, mask = [tokenizer.bos_id], [-100]
                for message in item["messages"]:
                    role = tokenizer.sp.piece_to_id(f"<|{message['role']}|>")
                    end = tokenizer.sp.piece_to_id("<|end|>")
                    body = tokenizer.encode(message["content"])
                    ids += [role, *body, end]
                    mask += [-100, *([*body, end] if message["role"] == "assistant" else [-100] * (len(body) + 1))]
                if len(ids) > 1 and len(ids) <= 1025 and any(x != -100 for x in mask[1:]):
                    group = item.get("task", "rules")
                    group = group if group in self.groups else "rules"
                    self.groups[group].append(len(self.items))
                    self.items.append((ids, mask))
        if not self.items:
            raise ValueError("No chat samples fit the context")
        _, _, desired = data_profile(root)
        self.groups = {key: value for key, value in self.groups.items() if value}
        self.weights = np.asarray([desired[key] for key in self.groups])
        self.weights /= self.weights.sum()
        self.pending_ids, self.pending_labels = [], []
        self.pending_group = None

    def state_dict(self):
        return {
            "rng": self.rng.bit_generator.state,
            "pending_ids": self.pending_ids.copy(),
            "pending_labels": self.pending_labels.copy(),
            "pending_group": self.pending_group,
        }

    def load_state_dict(self, state):
        self.rng.bit_generator.state = state["rng"]
        self.pending_ids = state["pending_ids"].copy()
        self.pending_labels = state["pending_labels"].copy()
        self.pending_group = state["pending_group"]

    def batch(self, size, length, group=None):
        if group != self.pending_group:
            self.pending_ids, self.pending_labels = [], []
            self.pending_group = group
        x, y = [], []
        for _ in range(size):
            while len(self.pending_ids) < length + 1:
                selected = group or self.rng.choice(list(self.groups), p=self.weights)
                index = self.rng.choice(self.groups[selected])
                ids, labels = self.items[index]
                self.pending_ids.extend(ids)
                self.pending_labels.extend(labels)
            x.append(self.pending_ids[:length])
            y.append(self.pending_labels[1 : length + 1])
            del self.pending_ids[:length]
            del self.pending_labels[:length]
        return np.asarray(x, dtype=np.int64), np.asarray(y, dtype=np.int64)


@lru_cache(maxsize=8)
def validation_documents(root, split="val", count=64):
    result = {}
    rng = random.Random(41000)
    for domain in WEIGHTS:
        reservoir = []
        with (Path(root) / f"{domain}.{split}.jsonl").open(encoding="utf-8") as reader:
            for index, line in enumerate(reader):
                text = json.loads(line)["text"]
                if index < count:
                    reservoir.append(text)
                else:
                    slot = rng.randrange(index + 1)
                    if slot < count:
                        reservoir[slot] = text
        if not reservoir:
            raise ValueError(f"Empty {domain} {split}")
        result[domain] = reservoir
    return result


@torch.inference_mode()
def evaluate(model, tokenizer, root, split="val", count=64, context_length=None):
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    context = context_length or model.cfg.context_length
    domains = {}
    for domain, documents in validation_documents(str(root), split, count).items():
        scores, document_characters, nll_sum, characters = [], [], 0.0, 0
        for text in documents:
            ids = tokenizer.encode(text, add_bos=True, add_eos=True)
            total_loss = 0.0
            for start in range(0, len(ids) - 1, context):
                window = torch.tensor(ids[start : start + context + 1], device=device)[None, :]
                with precision_context(device):
                    logits = model(window[:, :-1]).logits
                total_loss += float(
                    F.cross_entropy(
                        logits.float().reshape(-1, tokenizer.vocab_size), window[:, 1:].reshape(-1), reduction="sum"
                    )
                )
            chars = max(1, len(text))
            nll_sum += total_loss
            characters += chars
            scores.append(total_loss / chars / math.log(2))
            document_characters.append(chars)
        domains[domain] = {
            "bpc": nll_sum / characters / math.log(2),
            "document_bpc": scores,
            "characters": characters,
            "documents": len(scores),
            "document_characters": document_characters,
        }
    model.train(was_training)
    return {
        "macro_bpc": float(np.mean([x["bpc"] for x in domains.values()])),
        "domains": domains,
        "split": split,
        "context_length": context,
    }


@torch.inference_mode()
def evaluate_training(model, tokenizer, args):
    metrics = evaluate(model, tokenizer, args.data, count=args.eval_documents)
    manifest, _, _ = data_profile(args.data)
    story_objective = manifest.get("objective") == "story_continuation"
    if not args.chat:
        metrics["selection_score"] = metrics["domains"]["story"]["bpc"] if story_objective else metrics["macro_bpc"]
        metrics["selection_metric"] = "story_bpc" if story_objective else "macro_bpc"
        return metrics
    validation = ChatSampler(args.data, tokenizer, 72000, split="val")
    was_training = model.training
    model.eval()
    losses = {}
    device = next(model.parameters()).device
    for group in validation.groups:
        total, supervised = 0.0, 0
        for _ in range(8):
            x, y = validation.batch(1, model.cfg.context_length, group=group)
            x, y = torch.as_tensor(x, device=device), torch.as_tensor(y, device=device)
            with precision_context(device):
                logits = model(x).logits
            total += float(
                F.cross_entropy(
                    logits.float().reshape(-1, tokenizer.vocab_size), y.reshape(-1), ignore_index=-100, reduction="sum"
                )
            )
            supervised += int((y != -100).sum())
        losses[group] = total / supervised
    model.train(was_training)
    metrics.update(
        {
            "assistant_loss_by_task": losses,
            "selection_score": losses["story"] if story_objective else float(np.mean(list(losses.values()))),
            "selection_metric": "story_assistant_nll" if story_objective else "macro_assistant_nll",
        }
    )
    return metrics


def train(args, deadline=None):
    resources = []
    try:
        with attention_context(args.attention_backend):
            return _train(args, deadline, resources)
    finally:
        for resource in resources:
            if hasattr(resource, "close"):
                resource.close()


def _train(args, deadline, resources):
    deadline = deadline or Deadline(args.deadline_unix, args.max_seconds)
    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, deadline.request_stop)
    if deadline.expired():
        return {"status": "budget_stop", "tokens_seen": 0}
    seed_all(args.seed)
    device = torch.device(args.device)
    tokenizer = StoryTokenizer(args.data / "tokenizer.model")
    manifest_hash = blake2b_file(args.data / "manifest.json")
    data_manifest, _, _ = data_profile(args.data)
    tokenizer_hash = blake2b_file(args.data / "tokenizer.model")
    teacher = None
    if args.teacher:
        teacher, teacher_checkpoint = load_model(args.teacher, device)
        if teacher_checkpoint["tokenizer_blake2b16"] != tokenizer_hash:
            raise ValueError("Teacher vocabulary differs")
        teacher.eval().requires_grad_(False)
    sampler = (
        ChatSampler(args.data, tokenizer, args.seed + 40000)
        if args.chat
        else MixtureSampler(args.data, args.seed + 40000)
    )
    resources.append(sampler)
    latest = args.output / "latest.pt"
    if latest.exists():
        model, checkpoint = load_model(latest, device)
    elif args.initialize:
        model, checkpoint = load_model(args.initialize, device)
        if checkpoint["tokenizer_blake2b16"] != tokenizer_hash:
            raise ValueError("Initialization tokenizer differs")
        checkpoint = None
        if args.grow_teacher:
            model = grow_teacher(model)
    elif teacher is not None:
        model, checkpoint = student_from_teacher(teacher), None
    else:
        cfg = CANDIDATES[args.candidate]
        if tokenizer.vocab_size != cfg.vocab_size:
            raise ValueError("Tokenizer must be exactly 12000 pieces")
        model, checkpoint = DenseLanguageModel(cfg, build_surface_feature_table(tokenizer)).to(device), None
    if not args.allow_large and parameter_count(model.cfg) >= 18_000_000:
        raise ValueError("Student exceeds the strict 18M ceiling")
    if args.effective_tokens % (args.microbatch * model.cfg.context_length):
        raise ValueError("microbatch must divide the effective token batch")
    accum = args.effective_tokens // (args.microbatch * model.cfg.context_length)
    decay, no_decay = [], []
    for parameter in model.parameters():
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0}],
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=device.type == "cuda",
    )
    training_config = {
        "phase": args.phase,
        "candidate": args.candidate,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "schedule_tokens": args.schedule_tokens,
        "effective_tokens": args.effective_tokens,
        "chat": args.chat,
        "teacher_blake2b16": blake2b_file(args.teacher) if args.teacher else None,
        "data_order_version": 2,
        "chat_packing_version": 2 if args.chat else None,
    }
    step, tokens, best_bpc = 0, 0, float("inf")
    if checkpoint is not None:
        if checkpoint["training_config"] != training_config:
            raise ValueError("Training phase/schedule/teacher changed on resume")
        if (
            checkpoint["data_manifest_blake2b16"] != manifest_hash
            or checkpoint["tokenizer_blake2b16"] != tokenizer_hash
        ):
            raise ValueError("Training data or tokenizer changed on resume")
        optimizer.load_state_dict(checkpoint["optimizer"])
        sampler.load_state_dict(checkpoint["sampler"])
        random.setstate(checkpoint["python_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if device.type == "cuda" and checkpoint.get("cuda_rng"):
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        step, tokens, best_bpc = checkpoint["step"], checkpoint["tokens_seen"], checkpoint["best_bpc"]
        best_path = args.output / "best.pt"
        if best_path.exists():
            previous_best = torch.load(best_path, map_location="cpu", weights_only=False)
            if previous_best["training_config"] != training_config:
                raise ValueError("Best checkpoint belongs to a different training phase")
            best_bpc = min(best_bpc, previous_best["best_bpc"])
            del previous_best
    metadata = {
        "source_commit": source_commit(),
        "objective": data_manifest.get("objective", "general_korean"),
        "tokenizer_blake2b16": tokenizer_hash,
        "data_manifest_blake2b16": manifest_hash,
        "phase": args.phase,
        "teacher_path": str(args.teacher) if args.teacher else None,
        "chat_template": CHAT_TEMPLATE if args.chat else None,
        "execution": {
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "torch": torch.__version__,
            "attention": args.attention_backend,
            "loss": args.loss_backend if teacher is None else "torch_kd",
            "compiled": args.compile,
            "microbatch": args.microbatch,
        },
    }
    run_model = torch.compile(model, dynamic=True) if args.compile else model
    model.train()
    args.output.mkdir(parents=True, exist_ok=True)
    last_save, last_eval = time.monotonic(), tokens
    started, initial_tokens = time.monotonic(), tokens
    metrics = None
    while tokens < args.target_tokens and not deadline.expired():
        optimizer.zero_grad(set_to_none=True)
        warmup = min(2_000_000, max(args.effective_tokens, args.schedule_tokens * 0.01))
        progress = min(1.0, max(0.0, (tokens - warmup) / max(1, args.schedule_tokens - warmup)))
        rate = args.learning_rate * (
            min(1.0, (tokens + args.effective_tokens) / warmup)
            if tokens < warmup
            else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
        )
        for group in optimizer.param_groups:
            group["lr"] = rate
        total_loss = 0.0
        for _ in range(accum):
            x, y = sampler.batch(args.microbatch, model.cfg.context_length)
            x, y = torch.as_tensor(x, device=device), torch.as_tensor(y, device=device)
            with precision_context(device):
                if teacher is None:
                    loss = run_model(x, targets=y, loss_only=True, loss_backend=args.loss_backend).loss
                else:
                    logits = run_model(x).logits
                    with torch.no_grad():
                        teacher_logits = teacher(x).logits
                    loss = distillation_loss(logits, teacher_logits, y)
            (loss / accum).backward()
            total_loss += float(loss.detach()) / accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        step += 1
        tokens += args.effective_tokens
        if step % 10 == 0:
            print(
                json.dumps(
                    {
                        "phase": args.phase,
                        "step": step,
                        "tokens": tokens,
                        "loss": total_loss,
                        "lr": rate,
                        "tokens_per_second": (tokens - initial_tokens) / (time.monotonic() - started),
                    }
                ),
                flush=True,
            )
        if tokens - last_eval >= args.eval_tokens and not deadline.expired():
            metrics = evaluate_training(model, tokenizer, args)
            last_eval = tokens
            metrics = {**metrics, "step": step, "tokens_seen": tokens}
            atomic_json(args.output / "validation.json", metrics)
            if metrics["selection_score"] < best_bpc:
                best_bpc = metrics["selection_score"]
                save_checkpoint(
                    args.output / "best.pt",
                    model,
                    optimizer,
                    sampler,
                    step,
                    tokens,
                    best_bpc,
                    training_config,
                    {**metadata, "selection_validation": metrics},
                )
                atomic_json(args.output / "validation_best.json", metrics)
        if time.monotonic() - last_save >= 300 or deadline.expired():
            save_checkpoint(latest, model, optimizer, sampler, step, tokens, best_bpc, training_config, metadata)
            last_save = time.monotonic()
    save_checkpoint(latest, model, optimizer, sampler, step, tokens, best_bpc, training_config, metadata)
    if not deadline.expired():
        metrics = evaluate_training(model, tokenizer, args)
        metrics = {**metrics, "step": step, "tokens_seen": tokens}
        atomic_json(args.output / "validation.json", metrics)
        if metrics["selection_score"] < best_bpc:
            best_bpc = metrics["selection_score"]
            save_checkpoint(
                args.output / "best.pt",
                model,
                optimizer,
                sampler,
                step,
                tokens,
                best_bpc,
                training_config,
                {**metadata, "selection_validation": metrics},
            )
            atomic_json(args.output / "validation_best.json", metrics)
        save_checkpoint(latest, model, optimizer, sampler, step, tokens, best_bpc, training_config, metadata)
    result = {
        "status": "budget_stop" if deadline.expired() else "complete",
        "step": step,
        "tokens_seen": tokens,
        "best_bpc": best_bpc,
        "latest": str(latest),
        "validation": metrics,
    }
    atomic_json(args.output / "status.json", result)
    return result


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data", type=Path, default=Path("packed/haru-v3"))
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--candidate", choices=list(CANDIDATES), default="dense8")
    result.add_argument("--phase", default="candidate")
    result.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    result.add_argument("--microbatch", type=int, default=8)
    result.add_argument("--effective-tokens", type=int, default=131072)
    result.add_argument("--target-tokens", type=int, default=20_000_000)
    result.add_argument("--schedule-tokens", type=int, default=100_000_000)
    result.add_argument("--learning-rate", type=float, default=3e-4)
    result.add_argument("--seed", type=int, default=1337)
    result.add_argument("--initialize", type=Path)
    result.add_argument("--teacher", type=Path)
    result.add_argument("--grow-teacher", action="store_true")
    result.add_argument("--allow-large", action="store_true")
    result.add_argument("--chat", action="store_true")
    result.add_argument("--compile", action="store_true")
    result.add_argument("--attention-backend", choices=("auto", "flash", "cudnn", "math"), default="auto")
    result.add_argument("--loss-backend", choices=("torch", "liger"), default="torch")
    result.add_argument("--eval-tokens", type=int, default=5_000_000)
    result.add_argument("--eval-documents", type=int, default=64)
    result.add_argument("--deadline-unix", type=float)
    result.add_argument("--max-seconds", type=float)
    return result


def main():
    print(json.dumps(train(parser().parse_args()), ensure_ascii=False))


if __name__ == "__main__":
    main()
