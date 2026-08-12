from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Split = Literal["train", "validation"]

RELATION_CATEGORIES = ("location", "state", "ownership", "transfer", "speaker")

TRAIN_ENTITIES = (
    "가람",
    "나래",
    "다온",
    "라온",
    "마루",
    "보라",
    "새봄",
    "아라",
    "여울",
    "온유",
    "우람",
    "이든",
    "자람",
    "초롱",
    "푸름",
    "하람",
    "해솔",
    "도윤",
    "민서",
    "서준",
    "예린",
    "지후",
    "태윤",
    "현서",
)

VALIDATION_ENTITIES = ("겨울", "누리", "미르", "바다", "소담", "윤슬", "재희", "찬우", "하늬", "해나")

ATTRIBUTE_VALUES = {
    "location": (
        ("도서관", "정원"),
        ("주방", "옥상"),
        ("강변", "광장"),
        ("교실", "운동장"),
        ("동굴", "숲길"),
        ("역", "박물관"),
    ),
    "state": (
        ("젖어 있음", "말라 있음"),
        ("열려 있음", "닫혀 있음"),
        ("따뜻함", "차가움"),
        ("켜져 있음", "꺼져 있음"),
        ("깨끗함", "흐림"),
        ("가득 참", "비어 있음"),
    ),
}

PAYLOAD_PAIRS = {
    "ownership": (
        ("은색 열쇠", "구리 열쇠"),
        ("파란 우산", "노란 우산"),
        ("작은 수첩", "두꺼운 수첩"),
        ("별 배지", "달 배지"),
        ("유리 구슬", "나무 구슬"),
    ),
    "transfer": (
        ("봉인된 편지", "전달"),
        ("빨간 공", "건넴"),
        ("지도 한 장", "인계"),
        ("작은 상자", "전달"),
        ("물병", "건넴"),
    ),
    "speaker": (
        ("문은 이미 잠겼어", "창문은 열려 있어"),
        ("내일 다시 만나자", "오늘 바로 출발하자"),
        ("열쇠는 책상 아래 있어", "열쇠는 서랍 안에 있어"),
        ("나는 북쪽 길로 갈게", "나는 남쪽 길로 갈게"),
        ("상자는 비어 있어", "상자에는 편지가 있어"),
    ),
}

TRAIN_TEMPLATES = {
    "location": (
        "위치 기록: {a} = {x}; {b} = {y}. 질문: {a}의 위치는? 답:",
        "지도에는 {a}가 {x}에, {b}가 {y}에 있다고 표시되어 있다. {a}가 있는 곳은",
        "두 사람의 장소를 기억하자. {a}: {x}, {b}: {y}. 그러면 {a}의 장소는",
    ),
    "state": (
        "상태표: {a} = {x}; {b} = {y}. 질문: {a}의 상태는? 답:",
        "관찰 결과 {a}는 {x}, {b}는 {y} 상태다. {a}의 현재 상태는",
        "기록된 상태는 다음과 같다. {a}: {x}, {b}: {y}. {a}는 지금",
    ),
    "ownership": (
        "소유 기록: {a}가 {item}을 갖고 있고 {b}는 갖고 있지 않다. {item}의 주인은",
        "{item}은 {a}의 물건이며 {b}의 물건이 아니다. 질문: 누가 {item}을 소유하는가? 답:",
        "물품표에 {item}의 소유자가 {a}로 적혀 있다. {b}는 소유자가 아니다. 소유자는",
    ),
    "transfer": (
        "전달 기록: {a}가 {item}을 {b}에게 건넸다. {item}을 받은 사람은",
        "{a}에서 {b}로 {item}이 전달되었다. 질문: 수령자는 누구인가? 답:",
        "{a}는 보내는 사람이고 {b}는 {item}을 받는 사람이다. 받은 사람은",
    ),
    "speaker": (
        "대화 기록: {a}: ‘{quote}’ {b}: ‘{distractor}’ 질문: ‘{quote}’라고 말한 사람은",
        "{a}는 ‘{quote}’라고 했고 {b}는 ‘{distractor}’라고 했다. 첫 문장의 화자는",
        "발언표: ‘{quote}’ — {a}; ‘{distractor}’ — {b}. ‘{quote}’의 화자는",
    ),
}

VALIDATION_TEMPLATES = {
    "location": ("동선 보고서에 따르면 {a}의 도착지는 {x}, {b}의 도착지는 {y}다. {a}가 도착한 곳은",),
    "state": ("점검표에는 {a}가 {x}이고 {b}가 {y}라고 쓰여 있다. 점검 당시 {a}는",),
    "ownership": ("{item}의 명의자는 {a}로 확인되었고 {b}는 아니다. 명의자는",),
    "transfer": ("{item}의 이동 방향은 {a}에서 {b}다. 최종적으로 받은 사람은",),
    "speaker": ("녹취에는 {a}의 말이 ‘{quote}’, {b}의 말이 ‘{distractor}’로 남아 있다. ‘{quote}’의 발화자는",),
}


@dataclass(frozen=True, slots=True)
class CounterfactualPair:
    """Two minimally different prompts whose correct candidate must flip."""

    category: str
    template_id: str
    entity_a: str
    entity_b: str
    prompt_a: str
    prompt_b: str
    candidates: tuple[str, str]
    correct_indices: tuple[int, int] = (0, 1)


class CounterfactualSampler:
    """Create balanced relation pairs with fresh entity permutations per batch."""

    def __init__(self, split: Split = "train") -> None:
        if split not in {"train", "validation"}:
            raise ValueError(f"Unsupported split: {split}")
        self.split = split
        self.entities = TRAIN_ENTITIES if split == "train" else VALIDATION_ENTITIES
        self.templates = TRAIN_TEMPLATES if split == "train" else VALIDATION_TEMPLATES

    def sample_batch(self, pair_count: int, rng: np.random.Generator) -> tuple[CounterfactualPair, ...]:
        if pair_count <= 0:
            raise ValueError("pair_count must be positive")

        entity_order = self._sample_entity_order(pair_count * 2, rng)
        categories = [RELATION_CATEGORIES[index % len(RELATION_CATEGORIES)] for index in range(pair_count)]
        rng.shuffle(categories)

        pairs = []
        for index, category in enumerate(categories):
            entity_a = entity_order[index * 2]
            entity_b = entity_order[index * 2 + 1]
            pairs.append(self._render_pair(category, entity_a, entity_b, rng))
        return tuple(pairs)

    def _sample_entity_order(self, count: int, rng: np.random.Generator) -> list[str]:
        entities = []
        while len(entities) < count:
            permutation = rng.permutation(len(self.entities))
            entities.extend(self.entities[int(index)] for index in permutation)
        return entities[:count]

    def _render_pair(
        self,
        category: str,
        entity_a: str,
        entity_b: str,
        rng: np.random.Generator,
    ) -> CounterfactualPair:
        templates = self.templates[category]
        template_index = int(rng.integers(0, len(templates)))
        template = templates[template_index]
        template_id = f"{self.split}:{category}:{template_index}"

        if category in ATTRIBUTE_VALUES:
            values = ATTRIBUTE_VALUES[category]
            value_a, value_b = values[int(rng.integers(0, len(values)))]
            prompt_a = template.format(a=entity_a, b=entity_b, x=value_a, y=value_b)
            prompt_b = template.format(a=entity_a, b=entity_b, x=value_b, y=value_a)
            candidates = (value_a, value_b)
        else:
            payloads = PAYLOAD_PAIRS[category]
            payload, distractor = payloads[int(rng.integers(0, len(payloads)))]
            prompt_a = template.format(
                a=entity_a,
                b=entity_b,
                item=payload,
                quote=payload,
                distractor=distractor,
            )
            prompt_b = template.format(
                a=entity_b,
                b=entity_a,
                item=payload,
                quote=payload,
                distractor=distractor,
            )
            candidates = (entity_a, entity_b)
            if category == "transfer":
                candidates = (entity_b, entity_a)

        return CounterfactualPair(
            category=category,
            template_id=template_id,
            entity_a=entity_a,
            entity_b=entity_b,
            prompt_a=prompt_a,
            prompt_b=prompt_b,
            candidates=candidates,
        )
