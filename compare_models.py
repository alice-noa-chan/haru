"""Compare the released Haru, current Haru, and Tiny-Ko-Stories-35M."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer

from counterfactual_data import CounterfactualPair, CounterfactualSampler

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "runs" / "model_family_comparison.json"

MODEL_SPECS = {
    "haru_released_6.8m": {
        "directory": ROOT / "runs" / "haru" / "transformers",
        "parameters": 6_793_363,
        "training_tokens": 800_063_488,
        "architecture": "2 recurrent cells, 6 passes",
    },
    "haru_current_11.6m": {
        "directory": ROOT / "runs" / "haru-v2-binding" / "transformers" / "transformers",
        "parameters": 11_634_459,
        "training_tokens": 753_664_000,
        "architecture": "3 recurrent cells, 6 passes, final binding block",
    },
    "haru_v2_17m": {
        "directory": ROOT / "runs" / "haru-v2-unfolded" / "transformers",
        "parameters": 16_983_213,
        "training_tokens": 4_259_840_000,
        "architecture": "6 independent cells, 6 passes, final binding block",
    },
    "tiny_ko_stories_35m": {
        "directory": ROOT / "reference_models" / "Tiny-Ko-Stories-35M",
        "parameters": 34_217_856,
        "training_tokens": 794_820_608,
        "architecture": "10-layer full-attention Transformer",
    },
}

SHARED_TEXTS = [
    (
        "소라는 파란 구슬을 나무 상자에 넣고 상자를 책상 아래에 두었어요. "
        "동생이 빨간 구슬만 꺼낸 뒤 뚜껑을 닫았기 때문에 파란 구슬은 그대로 남아 있었어요."
    ),
    (
        "비가 그치자 민준은 젖은 우산을 현관에 펼쳐 두었어요. 창문으로 햇빛이 들어오자 "
        "우산의 물방울이 반짝였고, 민준은 바닥이 미끄럽지 않도록 수건으로 닦았어요."
    ),
    (
        "은지는 수현에게 빌린 책을 가방에 넣었어요. 다음 날 도서관에서 수현을 만나자 "
        "책을 돌려주며 재미있는 장면에 대해 한참 이야기했어요."
    ),
    (
        "작은 로봇은 밤마다 온실의 온도를 확인했어요. 어느 날 난방기가 멈추자 로봇은 "
        "관리인에게 신호를 보내고 어린 묘목 위에 보온 덮개를 씌웠어요."
    ),
    (
        "산길에서 길을 잃은 여우는 멀리서 들리는 종소리를 따라갔어요. 종소리는 양 떼의 "
        "목걸이에서 났고, 목동은 여우에게 마을로 내려가는 길을 알려 주었어요."
    ),
]

GENERATION_PROMPTS = [
    "서윤이는 금빛 단추를 유리병에 넣어 찬장 위에 두었어요. 민호에게 병을 건드리지 말라고 했지만, 잠시 뒤 부엌에서 쨍그랑 소리가 났어요. 서윤이가 달려가 보니",
    "토끼는 다리를 다쳐 경주에 나가지 않았고 거북이만 출발선에 섰어요. 종이 울리자",
    "“내일 네 책을 돌려줄게.” 은지가 말했어요. 다음 날 은지는 준호를 만나",
]


def build_cases() -> list[dict[str, str]]:
    names = ["지우", "수아", "민서", "유나", "지호", "서윤", "보리", "토리"]
    animals = ["강아지", "고양이", "토끼", "거북이", "여우", "곰", "오리", "다람쥐"]
    objects = ["사과", "편지", "열쇠", "단추", "모자"]
    colors = [("빨간", "파란"), ("노란", "초록"), ("하얀", "검은"), ("금빛", "은빛"), ("분홍", "보라")]
    cases: list[dict[str, str]] = []

    for index in range(10):
        name = names[index % len(names)]
        first_color, second_color = colors[index % len(colors)]
        target_first = index % 2 == 0
        target_color = first_color if target_first else second_color
        expected_side = "왼쪽" if target_first else "오른쪽"
        wrong_side = "오른쪽" if target_first else "왼쪽"
        cases.append(
            {
                "category": "location",
                "prompt": (
                    f"{name}는 {first_color} 열쇠를 왼쪽 서랍에 넣고 {second_color} 열쇠를 오른쪽 서랍에 넣었어요. "
                    f"잠시 뒤 {target_color} 열쇠가 필요해져서 {name}는"
                ),
                "expected": f" {expected_side} 서랍을 열었어요.",
                "contradiction": f" {wrong_side} 서랍을 열었어요.",
            }
        )

        giver = names[index % len(names)]
        receiver = names[(index + 3) % len(names)]
        cases.append(
            {
                "category": "transfer",
                "prompt": (
                    f"{giver}는 공을 {receiver}에게 건넸어요. {receiver}는 공을 꼭 안고 운동장 끝까지 달렸어요. "
                    "선생님이 지금 공을 가진 아이를 불렀어요. 그 아이는"
                ),
                "expected": f" {receiver}였어요.",
                "contradiction": f" {giver}였어요.",
            }
        )

        outside = animals[index % len(animals)]
        inside = animals[(index + 3) % len(animals)]
        cases.append(
            {
                "category": "negation",
                "prompt": (
                    f"비가 내리자 {inside}만 현관 안으로 들어왔어요. {outside}는 겁이 나서 마당에 남아 있었어요. "
                    "현관 바닥에 젖은 발자국을 남긴 동물은"
                ),
                "expected": f" {inside}였어요.",
                "contradiction": f" {outside}였어요.",
            }
        )

        pair_index = index // 2
        item = objects[pair_index]
        state_name = names[pair_index]
        item_present = index % 2 == 0
        state_change = (
            "그 뒤에는 아무도 상자를 열지 않았어요."
            if item_present
            else f"잠시 뒤 {state_name}는 {item}를 다시 꺼내고 빈 상자를 닫았어요."
        )
        cases.append(
            {
                "category": "persistent_state",
                "prompt": (
                    f"{state_name}는 빈 상자에 {item}를 넣고 뚜껑을 닫았어요. {state_change} "
                    "저녁이 되어 상자를 열어 보니 안에는"
                ),
                "expected": f" {item}가 {'있었어요' if item_present else '없었어요'}.",
                "contradiction": f" {item}가 {'없었어요' if item_present else '있었어요'}.",
            }
        )

        lender = names[(index + 1) % len(names)]
        borrower = names[(index + 5) % len(names)]
        cases.append(
            {
                "category": "speaker_role",
                "prompt": (
                    f"“내 우산을 빌려줄게.” {lender}가 말했어요. {borrower}는 {lender}의 우산을 받아 비를 피했어요. "
                    "다음 날 우산을 돌려주며 고맙다고 말한 사람은"
                ),
                "expected": f" {borrower}였어요.",
                "contradiction": f" {lender}였어요.",
            }
        )

    if len(cases) != 50:
        raise RuntimeError(f"Expected 50 cases, got {len(cases)}")
    return cases


class HaruRunner:
    def __init__(self, directory: Path) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(directory, trust_remote_code=True, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            directory, trust_remote_code=True, local_files_only=True
        ).eval()
        self.model.config.inference_recurrences = 6

    def completion_score(self, prompt: str, completion: str) -> tuple[float, int]:
        prompt_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
        completion_ids = self.tokenizer(completion, add_special_tokens=False, return_tensors="pt")["input_ids"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        with torch.inference_mode():
            logits = self.model(input_ids=input_ids, recurrences=6).logits
            start = prompt_ids.shape[1] - 1
            selected = torch.log_softmax(logits[:, start:-1], dim=-1).gather(-1, completion_ids.unsqueeze(-1))
        return float(selected.sum()), int(completion_ids.numel())

    def appended_completion_score(self, prompt: str, completion: str) -> tuple[float, int]:
        prefix_ids = self.tokenizer(prompt, add_special_tokens=True)["input_ids"]
        full_ids = self.tokenizer(prompt + completion, add_special_tokens=True)["input_ids"]
        if full_ids[: len(prefix_ids)] != prefix_ids:
            raise ValueError("Tokenizer changed the prompt prefix after appending a candidate")
        input_ids = torch.tensor([full_ids], dtype=torch.long)
        with torch.inference_mode():
            logits = self.model(input_ids=input_ids, recurrences=6).logits
            start = len(prefix_ids) - 1
            targets = input_ids[:, len(prefix_ids) :]
            selected = torch.log_softmax(logits[:, start:-1], dim=-1).gather(-1, targets.unsqueeze(-1))
        return float(selected.sum()), int(targets.numel())

    def text_nll(self, text: str) -> tuple[float, int]:
        input_ids = self.tokenizer(text, return_tensors="pt")["input_ids"]
        with torch.inference_mode():
            logits = self.model(input_ids=input_ids, recurrences=6).logits
            selected = torch.log_softmax(logits[:, :-1], dim=-1).gather(-1, input_ids[:, 1:].unsqueeze(-1))
        return float(-selected.sum()), int(input_ids.shape[1] - 1)

    def generate(self, prompt: str, seed: int) -> tuple[str, int, float]:
        torch.manual_seed(seed)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=True,
                temperature=0.55,
                top_p=0.9,
                top_k=40,
                repetition_penalty=1.08,
                use_cache=False,
                bad_words_ids=[
                    [self.tokenizer.pad_token_id],
                    [self.tokenizer.bos_token_id],
                    [self.tokenizer.unk_token_id],
                ],
            )
        elapsed = time.perf_counter() - started
        generated_ids = output[0, inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).lstrip()
        return text, int(generated_ids.numel()), elapsed


class ReferenceRunner:
    def __init__(self, directory: Path) -> None:
        module_path = directory / "model.py"
        spec = importlib.util.spec_from_file_location("tiny_ko_35m_comparison", module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load reference model module: {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        checkpoint = torch.load(directory / "tiny-ko-stories-35m.pt", map_location="cpu", weights_only=True)
        model_config = module.LMConfig.from_dict(checkpoint["model_config"])
        self.model = module.StoryLM(model_config)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        self.tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        self.bos_id = self.tokenizer.token_to_id("<bos>")
        self.eos_id = self.tokenizer.token_to_id("<eos>")

    def encode_prompt(self, prompt: str) -> list[int]:
        ids = self.tokenizer.encode(prompt, add_special_tokens=False).ids
        return ([self.bos_id] if self.bos_id is not None else []) + ids

    def completion_score(self, prompt: str, completion: str) -> tuple[float, int]:
        prompt_ids = self.encode_prompt(prompt)
        completion_ids = self.tokenizer.encode(completion, add_special_tokens=False).ids
        input_ids = torch.tensor([prompt_ids + completion_ids], dtype=torch.long)
        with torch.inference_mode():
            logits, _ = self.model(input_ids)
            start = len(prompt_ids) - 1
            targets = torch.tensor([completion_ids], dtype=torch.long)
            selected = torch.log_softmax(logits[:, start:-1], dim=-1).gather(-1, targets.unsqueeze(-1))
        return float(selected.sum()), len(completion_ids)

    def appended_completion_score(self, prompt: str, completion: str) -> tuple[float, int]:
        prefix_ids = self.encode_prompt(prompt)
        full_ids = self.encode_prompt(prompt + completion)
        if full_ids[: len(prefix_ids)] != prefix_ids:
            raise ValueError("Tokenizer changed the prompt prefix after appending a candidate")
        input_ids = torch.tensor([full_ids], dtype=torch.long)
        with torch.inference_mode():
            logits, _ = self.model(input_ids)
            start = len(prefix_ids) - 1
            targets = input_ids[:, len(prefix_ids) :]
            selected = torch.log_softmax(logits[:, start:-1], dim=-1).gather(-1, targets.unsqueeze(-1))
        return float(selected.sum()), int(targets.numel())

    def text_nll(self, text: str) -> tuple[float, int]:
        ids = self.encode_prompt(text)
        input_ids = torch.tensor([ids], dtype=torch.long)
        with torch.inference_mode():
            logits, _ = self.model(input_ids)
            selected = torch.log_softmax(logits[:, :-1], dim=-1).gather(-1, input_ids[:, 1:].unsqueeze(-1))
        return float(-selected.sum()), len(ids) - 1

    def generate(self, prompt: str, seed: int) -> tuple[str, int, float]:
        torch.manual_seed(seed)
        prompt_ids = self.encode_prompt(prompt)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long)
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(
                input_ids,
                max_new_tokens=80,
                temperature=0.55,
                top_p=0.9,
                top_k=40,
                repetition_penalty=1.08,
                eos_token_id=self.eos_id,
            )
        elapsed = time.perf_counter() - started
        generated_ids = output[0, len(prompt_ids) :].tolist()
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).lstrip()
        return text, len(generated_ids), elapsed


def summarize_rows(rows: list[dict[str, Any]], included_ids: set[int]) -> dict[str, Any]:
    selected = [row for row in rows if row["case_id"] in included_ids]
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        by_category[row["category"]].append(row)
    return {
        "correct": sum(row["correct"] for row in selected),
        "total": len(selected),
        "accuracy": sum(row["correct"] for row in selected) / len(selected),
        "mean_margin_nats": sum(row["margin_nats"] for row in selected) / len(selected),
        "by_category": {
            category: {
                "correct": sum(row["correct"] for row in category_rows),
                "total": len(category_rows),
            }
            for category, category_rows in sorted(by_category.items())
        },
    }


def build_counterfactual_validation_pairs() -> list[CounterfactualPair]:
    """Recreate the fixed 100-pair validation sequence used during training."""

    rng = np.random.default_rng(1_337 + 20_000)
    sampler = CounterfactualSampler("validation")
    pairs = []
    while len(pairs) < 100:
        pairs.extend(sampler.sample_batch(min(5, 100 - len(pairs)), rng))
    return pairs


def evaluate_counterfactual_pairs(
    runner: HaruRunner | ReferenceRunner, pairs: list[CounterfactualPair]
) -> dict[str, Any]:
    rows = []
    for pair_id, pair in enumerate(pairs):
        prompt_a = pair.prompt_a.rstrip() + "\n"
        prompt_b = pair.prompt_b.rstrip() + "\n"
        a_on_a, a_on_a_tokens = runner.appended_completion_score(prompt_a, pair.candidates[0])
        b_on_a, b_on_a_tokens = runner.appended_completion_score(prompt_a, pair.candidates[1])
        a_on_b, a_on_b_tokens = runner.appended_completion_score(prompt_b, pair.candidates[0])
        b_on_b, b_on_b_tokens = runner.appended_completion_score(prompt_b, pair.candidates[1])
        first_margin = a_on_a / a_on_a_tokens - b_on_a / b_on_a_tokens
        second_margin = b_on_b / b_on_b_tokens - a_on_b / a_on_b_tokens
        rows.append(
            {
                "pair_id": pair_id,
                "category": pair.category,
                "first_correct": first_margin > 0.0,
                "second_correct": second_margin > 0.0,
                "strict_correct": first_margin > 0.0 and second_margin > 0.0,
                "first_margin": first_margin,
                "second_margin": second_margin,
            }
        )

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)
    decisions_correct = sum(row["first_correct"] + row["second_correct"] for row in rows)
    strict_correct = sum(row["strict_correct"] for row in rows)
    return {
        "decision_correct": decisions_correct,
        "decision_total": 2 * len(rows),
        "decision_accuracy": decisions_correct / (2 * len(rows)),
        "strict_pair_correct": strict_correct,
        "strict_pair_total": len(rows),
        "strict_pair_accuracy": strict_correct / len(rows),
        "prediction_flip_count": sum(row["first_correct"] == row["second_correct"] for row in rows),
        "mean_margin": sum(row["first_margin"] + row["second_margin"] for row in rows) / (2 * len(rows)),
        "by_category": {
            category: {
                "decision_correct": sum(row["first_correct"] + row["second_correct"] for row in category_rows),
                "decision_total": 2 * len(category_rows),
                "strict_pair_correct": sum(row["strict_correct"] for row in category_rows),
                "strict_pair_total": len(category_rows),
            }
            for category, category_rows in sorted(by_category.items())
        },
        "rows": rows,
    }


def compare(output_path: Path) -> dict[str, Any]:
    torch.set_num_threads(8)
    cases = build_cases()
    counterfactual_pairs = build_counterfactual_validation_pairs()
    runners: dict[str, HaruRunner | ReferenceRunner] = {
        # Adding an entry to MODEL_SPECS is not enough on its own; a model has
        # to be listed here too, and a missing one is silent. v2.0 was added
        # above and simply did not appear in the results.
        "haru_released_6.8m": HaruRunner(MODEL_SPECS["haru_released_6.8m"]["directory"]),
        "haru_current_11.6m": HaruRunner(MODEL_SPECS["haru_current_11.6m"]["directory"]),
        "haru_v2_17m": HaruRunner(MODEL_SPECS["haru_v2_17m"]["directory"]),
        "tiny_ko_stories_35m": ReferenceRunner(MODEL_SPECS["tiny_ko_stories_35m"]["directory"]),
    }

    result: dict[str, Any] = {
        "settings": {
            "device": "cpu",
            "threads": 8,
            "forced_choice_cases": len(cases),
            "generation": {
                "max_new_tokens": 80,
                "temperature": 0.55,
                "top_p": 0.9,
                "top_k": 40,
                "repetition_penalty": 1.08,
                "seed": 42,
            },
        },
        "models": {},
    }
    common_case_ids = set(range(len(cases)))

    missing = sorted(set(MODEL_SPECS) - set(runners))
    if missing:
        raise ValueError(f"MODEL_SPECS lists {missing} but no runner was built for them")

    for name, runner in runners.items():
        started = time.perf_counter()
        rows = []
        matched_ids = set()
        for case_id, case in enumerate(cases):
            expected_score, expected_tokens = runner.completion_score(case["prompt"], case["expected"])
            wrong_score, wrong_tokens = runner.completion_score(case["prompt"], case["contradiction"])
            token_matched = expected_tokens == wrong_tokens
            if token_matched:
                matched_ids.add(case_id)
            rows.append(
                {
                    "case_id": case_id,
                    "category": case["category"],
                    "token_matched": token_matched,
                    "expected_tokens": expected_tokens,
                    "contradiction_tokens": wrong_tokens,
                    "correct": expected_score > wrong_score,
                    "margin_nats": expected_score - wrong_score,
                }
            )
        common_case_ids &= matched_ids

        total_nll = 0.0
        total_tokens = 0
        total_characters = 0
        for text in SHARED_TEXTS:
            nll, tokens = runner.text_nll(text)
            total_nll += nll
            total_tokens += tokens
            total_characters += len(text)

        generations = []
        for prompt in GENERATION_PROMPTS:
            continuation, tokens, seconds = runner.generate(prompt, seed=42)
            generations.append(
                {
                    "prompt": prompt,
                    "continuation": continuation,
                    "generated_tokens": tokens,
                    "continuation_characters": len(continuation),
                    "seconds": seconds,
                    "tokens_per_second": tokens / seconds,
                    "characters_per_second": len(continuation) / seconds,
                }
            )

        result["models"][name] = {
            **{key: value for key, value in MODEL_SPECS[name].items() if key != "directory"},
            "matched_case_ids": sorted(matched_ids),
            "forced_choice_rows": rows,
            "shared_text": {
                "texts": len(SHARED_TEXTS),
                "characters": total_characters,
                "tokens": total_tokens,
                "nll_nats": total_nll,
                "bits_per_character": total_nll / (total_characters * math.log(2)),
            },
            "counterfactual_validation": evaluate_counterfactual_pairs(runner, counterfactual_pairs),
            "generations": generations,
            "elapsed_seconds": time.perf_counter() - started,
        }

    for model_result in result["models"].values():
        model_result["forced_choice_common"] = summarize_rows(model_result["forced_choice_rows"], common_case_ids)
    result["common_case_ids"] = sorted(common_case_ids)
    result["common_case_count"] = len(common_case_ids)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = compare(args.output)
    summary = {
        name: {
            "parameters": model["parameters"],
            "training_tokens": model["training_tokens"],
            "forced_choice": model["forced_choice_common"],
            "counterfactual_validation": {
                key: value
                for key, value in model["counterfactual_validation"].items()
                if key not in {"rows", "by_category"}
            },
            "bits_per_character": model["shared_text"]["bits_per_character"],
            "generation_characters_per_second": sum(row["continuation_characters"] for row in model["generations"])
            / sum(row["seconds"] for row in model["generations"]),
        }
        for name, model in result["models"].items()
    }
    print(json.dumps({"output": str(args.output), "common_cases": result["common_case_count"], **summary}, indent=2))


if __name__ == "__main__":
    main()
