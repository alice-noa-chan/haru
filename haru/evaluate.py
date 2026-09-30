"""Final-only raw-text BPC, relation generalization and CPU generation comparison.

Uses identical raw documents at context 512 for legacy and dense models, plus
the dense model's native 1024 context. Public KoBEST is a separate final-only
command in evaluate_korean.py and never drives pipeline selection.
"""

from __future__ import annotations

import argparse
import gc
import re
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from compare_models import GENERATION_PROMPTS, build_cases
from haru.data import rule_examples
from haru.runtime import atomic_json
from haru.training import evaluate
from tokenizer_utils import StoryTokenizer


class ScoringModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.wrapper = model
        self.cfg = SimpleNamespace(context_length=model.config.max_position_embeddings)

    def forward(self, ids):
        return self.wrapper(ids, use_cache=False)


@torch.inference_mode()
def completion_score(model, tokenizer, prompt, answer):
    prefix = tokenizer.encode(prompt, add_special_tokens=True)
    suffix = tokenizer.encode(answer, add_special_tokens=False)
    ids = torch.tensor([prefix + suffix], device=next(model.parameters()).device)
    logits = model(ids, use_cache=False).logits[:, len(prefix) - 1 : -1].float()
    targets = ids[:, len(prefix) :]
    score = logits.log_softmax(-1).gather(-1, targets.unsqueeze(-1)).sum().item()
    return score


def normalize_answer(text):
    return re.sub(r"[\s.!?。]+", "", text)


@torch.inference_mode()
def diagnose(directory, data, documents=64, rules=70, threads=4):
    torch.set_num_threads(threads)
    tokenizer = AutoTokenizer.from_pretrained(directory, trust_remote_code=True, local_files_only=True)
    model = (
        AutoModelForCausalLM.from_pretrained(directory, trust_remote_code=True, local_files_only=True).float().eval()
    )
    dense = model.config.model_type == "haru_dense"
    eos_ids = model.generation_config.eos_token_id
    chat = dense and isinstance(eos_ids, list) and tokenizer.convert_tokens_to_ids("<|end|>") in eos_ids
    context = model.config.max_position_embeddings
    story_tokenizer = StoryTokenizer(Path(directory) / "tokenizer.model")
    result = {
        "directory": str(directory),
        "parameters": sum(p.numel() for p in model.parameters()),
        "model_type": model.config.model_type,
        "cpu_threads": threads,
        "bpc_shared_context": evaluate(
            ScoringModel(model), story_tokenizer, data, split="test", count=documents, context_length=512
        ),
    }
    if dense:
        result["bpc_native_context"] = evaluate(
            ScoringModel(model), story_tokenizer, data, split="test", count=documents
        )
    relations = defaultdict(list)
    for case in build_cases():
        good = completion_score(model, tokenizer, case["prompt"], case["expected"])
        bad = completion_score(model, tokenizer, case["prompt"], case["contradiction"])
        relations[case["category"]].append(good > bad)
    result["legacy_relation_ranking"] = {
        task: {"accuracy": float(np.mean(values)), "count": len(values)} for task, values in relations.items()
    }
    result["program_generalization"] = {}
    for split, forms in (("train", "train"), ("test", "train"), ("train", "test"), ("test", "test")):
        rows = defaultdict(list)
        for item in rule_examples(rules, split, seed=71833, template_split=forms):
            prompt = item["prompt"]
            if chat:
                inputs = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                )
            else:
                inputs = tokenizer(prompt, return_tensors="pt")
            ids = model.generate(**inputs, max_new_tokens=12, do_sample=False, use_cache=dense)
            answer = tokenizer.decode(ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
            # A leading correct answer followed by unrelated text is not an exact match.
            rows[item["task"]].append(normalize_answer(answer) == normalize_answer(item["answer"]))
        key = (
            f"{'known' if split == 'train' else 'novel'}_entities_{'known' if forms == 'train' else 'novel'}_templates"
        )
        result["program_generalization"][key] = {
            task: {"exact_match": float(np.mean(values)), "count": len(values)} for task, values in rows.items()
        }
    generation = []
    for prompt in GENERATION_PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt")
        samples = []
        for _ in range(4):
            started = time.perf_counter()
            model(**inputs, use_cache=dense)
            prefill = time.perf_counter() - started
            started = time.perf_counter()
            output = model.generate(**inputs, max_new_tokens=80, do_sample=False, use_cache=dense)
            samples.append((prefill, time.perf_counter() - started))
        generated = output[0, inputs["input_ids"].shape[1] :].tolist()
        grams = [tuple(generated[i : i + 4]) for i in range(max(0, len(generated) - 3))]
        generation.append(
            {
                "prompt": prompt,
                "completion": tokenizer.decode(generated, skip_special_tokens=True),
                "tokens": len(generated),
                "prefill_seconds": float(np.median([s[0] for s in samples[1:]])),
                "tokens_per_second": len(generated) / float(np.median([s[1] for s in samples[1:]])),
                "repeated_4gram_fraction": 1 - len(set(grams)) / len(grams) if grams else 0.0,
                "fact_retention": "requires human review of the saved prompt and continuation",
            }
        )
    result["generation"] = generation
    if chat:
        result["chat_generation"] = []
        for prompt in GENERATION_PROMPTS:
            content = "다음 동화의 인물, 물건, 사건을 유지하며 이어 써 주세요.\n\n" + prompt
            inputs = tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
            output = model.generate(**inputs, max_new_tokens=80, do_sample=False, use_cache=True)
            generated = output[0, inputs["input_ids"].shape[1] :].tolist()
            result["chat_generation"].append(
                {
                    "prompt": content,
                    "completion": tokenizer.decode(generated, skip_special_tokens=True),
                    "tokens": len(generated),
                    "fact_retention": "requires human review of the saved prompt and continuation",
                }
            )
    result["native_context_length"] = context
    result["limitations"] = (
        "Small synthetic exact-match and ranking tests; no claim of universal reasoning. Fact retention requires human review."
    )
    del model
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, action="append", required=True)
    parser.add_argument("--data", type=Path, default=Path("packed/haru-v3"))
    parser.add_argument("--documents", type=int, default=64)
    parser.add_argument("--rules", type=int, default=70)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("results/haru_dense_final.json"))
    args = parser.parse_args()
    report = {"split": "test", "models": []}
    for directory in args.model:
        report["models"].append(diagnose(directory, args.data, args.documents, args.rules, args.threads))
        atomic_json(args.output, report)


if __name__ == "__main__":
    main()
