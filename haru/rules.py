"""Ground-truth state transitions with disjoint Korean rendering families."""

from __future__ import annotations

import random


def examples(count, split="train", seed=1337, names_split=None, template_split=None):
    from haru.data import LOCATIONS, NAMES, OBJECTS, PREFIXES, particle

    rng = random.Random(seed)
    forms = template_split or split
    families = {"train": (0, 1, 2), "val": (3, 4), "test": (5, 6)}
    categories = ("location", "state", "ownership", "transfer", "speaker", "negation", "arithmetic")
    for index in range(count):
        people = rng.sample(NAMES[names_split or split], 3)
        obj = rng.choice(OBJECTS[split])
        places = rng.sample(LOCATIONS[split], 2)
        family = rng.choice(families[forms])
        task = categories[index % len(categories)]
        first, second, third = people

        def topic(name):
            return particle(name, "은", "는")

        accusative = particle(obj, "을", "를")
        # Val/test also contain longer event chains than training.
        event_count = rng.randint(1, 3) if forms == "train" else rng.randint(2, 4)
        order = []
        if task == "location":
            location = rng.choice(places)
            facts = [f"처음 {obj}의 위치: {location}."]
            for _ in range(event_count):
                actor, location = rng.choice(people), rng.choice(places)
                order.append(places.index(location))
                facts.append(
                    [
                        f"{topic(actor)} {accusative} {location}에 옮겼습니다.",
                        f"{actor}의 이동으로 {obj}이 놓인 곳은 {location}입니다.",
                        f"이어서 {actor}가 {location}에 {accusative} 넣었습니다.",
                        f"다음 기록에는 {actor}가 {obj}의 위치를 {location}로 변경했다고 적혀 있습니다.",
                        f"{location} 안으로 {obj}이 이동한 것은 {actor}의 행동 때문입니다.",
                        f"그 후 {actor}에 의해 {obj}의 보관 장소가 {location}로 바뀌었습니다.",
                        f"뒤따른 사건: {actor}가 {obj}의 새 위치를 {location}로 정했습니다.",
                    ][family]
                )
            answer = location
            question = [
                f"지금 {obj}은 어디에 있나요?",
                f"{obj}의 현재 위치는?",
                f"마지막 {obj}의 장소를 쓰세요.",
                f"이동이 끝난 시점에 {obj}을 찾을 곳을 답하세요.",
                f"모든 사건 이후 {obj}의 보관처는 어디입니까?",
                f"가장 나중의 기록을 반영하면 {obj}을 어느 장소에서 발견할까요?",
                f"위 기록이 종료된 뒤 {obj}의 소재지를 적어 주세요.",
            ][family]
        elif task == "state":
            opened = bool(rng.getrandbits(1))
            facts = [f"시작할 때 문은 {'열린' if opened else '닫힌'} 상태였습니다."]
            for _ in range(event_count):
                actor, opened = rng.choice(people), bool(rng.getrandbits(1))
                action = "열었습니다" if opened else "닫았습니다"
                status = "열림" if opened else "닫힘"
                order.append(int(opened))
                facts.append(
                    [
                        f"{topic(actor)} 문을 {action}.",
                        f"{actor}가 문을 {'열어' if opened else '닫아'} {'열린' if opened else '닫힌'} 상태로 만들었습니다.",
                        f"그 다음 문의 상태를 {status}으로 바꾼 사람은 {actor}입니다.",
                        f"뒤이어 문에 {actor}가 가한 조작은 {status}입니다.",
                        f"다음 사건에서 {actor}의 조작 결과는 문 {status}이었습니다.",
                        f"이후 {actor}의 행동이 완료되면서 문은 {status} 상태가 됐습니다.",
                        f"추가 기록의 문 상태 변경값은 {status}이며 행동한 사람은 {actor}입니다.",
                    ][family]
                )
            answer = "열린 상태" if opened else "닫힌 상태"
            question = [
                "현재 문은 어떤 상태인가요?",
                "문이 열려 있나요, 닫혀 있나요?",
                "마지막 문의 상태는?",
                "조작이 모두 끝난 문의 상태를 답하세요.",
                "이 기록의 최종 문 상태를 판단하세요.",
                "마지막 조작 이후 문은 열린 상태와 닫힌 상태 중 어느 쪽입니까?",
                "기록 종료 시점에서 관찰할 문의 상태를 적으세요.",
            ][family]
        elif task in ("ownership", "transfer"):
            owner = rng.choice(people)
            facts = [f"처음 {obj}의 소유자: {owner}."]
            for _ in range(event_count):
                receiver = rng.choice([person for person in people if person != owner])
                gift = task == "transfer" and bool(rng.getrandbits(1))
                order.append(int(gift))
                if gift:
                    facts.append(
                        [
                            f"{topic(owner)} {accusative} {receiver}에게 선물했습니다.",
                            f"{receiver}가 {owner}에게서 {accusative} 선물받았습니다.",
                            f"{obj}의 소유권을 {owner}가 {receiver}에게 넘겼습니다.",
                            f"그 다음 {owner}의 {obj} 소유권이 {receiver}에게 양도됐습니다.",
                            f"이후 증여가 이뤄져 {obj}의 새 주인은 {receiver}가 됐습니다.",
                            f"이어진 소유권 변경 기록은 {owner}에서 {receiver}로의 증여입니다.",
                            f"추가 사건에서 {owner}가 자신의 {accusative} {receiver}의 소유로 만들었습니다.",
                        ][family]
                    )
                    owner = receiver
                else:
                    facts.append(
                        [
                            f"{topic(receiver)} {accusative} 빌렸습니다. 주인은 바뀌지 않았습니다.",
                            f"{receiver}에게 {accusative} 빌려줬지만 소유권은 넘기지 않았습니다.",
                            f"{receiver}가 {accusative} 잠시 사용했습니다. 선물은 아니었습니다.",
                            f"뒤따른 대여에서 {receiver}는 사용권만 얻었으며 소유권 양도는 없었습니다.",
                            f"{receiver}의 임시 사용은 증여에 해당하지 않으므로 소유자는 그대로입니다.",
                            f"그 뒤 {receiver}가 보관을 맡았으나 주인이 된 것은 아닙니다.",
                            f"후속 사건은 {receiver}에게 빌려주는 것이었고 소유자 변경을 수반하지 않았습니다.",
                        ][family]
                    )
            answer = owner
            question = [
                f"마지막 {obj}의 주인은 누구인가요?",
                f"현재 {obj}의 소유자는?",
                f"{obj}의 최종 주인을 쓰세요.",
                f"이 사건들을 반영한 {obj}의 소유자 이름을 답하세요.",
                f"모든 대여와 증여 이후 {obj}의 소유권자는 누구입니까?",
                f"사용자와 주인을 구분했을 때 종료 시점의 {obj} 소유권은 누구에게 있습니까?",
                f"기록 종료 후 {obj}의 주인으로 남는 인물 이름을 적어 주세요.",
            ][family]
        elif task == "speaker":
            utterances = ["오늘은 맑아요", "내일 만나자", "문을 닫아 주세요"]
            rng.shuffle(people)
            target = rng.randrange(3)
            order = [target]
            facts = []
            for person, utterance in zip(people, utterances):
                facts.append(
                    [
                        f"{topic(person)} '{utterance}'라고 말했습니다.",
                        f"'{utterance}'라고 한 사람은 {person}입니다.",
                        f"{person}의 말: '{utterance}'.",
                        f"대화 기록에서 '{utterance}'라는 발언은 {person}에게 귀속됩니다.",
                        f"발언자 {person}의 문장은 '{utterance}'입니다.",
                        f"발화 기록은 인물 {person}와 발언 '{utterance}'를 연결합니다.",
                        f"'{utterance}'라는 말을 꺼낸 인물의 이름은 {particle(person, '이었습니다', '였습니다')}.",
                    ][family]
                )
            answer = people[target]
            question = [
                f"'{utterances[target]}'라고 말한 사람은?",
                f"누가 '{utterances[target]}'라고 했나요?",
                f"'{utterances[target]}'의 발언자는?",
                f"'{utterances[target]}'라는 발언에 해당하는 인물 이름을 답하세요.",
                f"'{utterances[target]}' 문장의 화자를 확인하세요.",
                f"주어진 대화에서 '{utterances[target]}'라는 발화를 한 인물의 이름을 적으세요.",
                f"'{utterances[target]}' 발언과 연결된 화자 이름을 찾아 주세요.",
            ][family]
        elif task == "negation":
            absent = rng.choice(people[:2])
            present = second if absent == first else first
            facts = [f"{topic(present)} {accusative} 가져왔습니다."]
            facts.append(
                [
                    f"{topic(absent)} {accusative} 가져오지 않았습니다.",
                    f"{topic(absent)} {obj}을 챙기지 않았습니다.",
                    f"{absent}에게는 가져온 {obj}이 없습니다.",
                    f"{absent}가 {accusative} 가져왔다는 주장은 사실이 아닙니다.",
                    f"{obj}을 지참한 인물 중에 {absent}는 포함되지 않습니다.",
                    f"{absent}에 관해서는 {obj}을 가져온 일이 없다고 기록돼 있습니다.",
                    f"{absent}가 {obj}을 지참했을 가능성은 기록에 의해 부정됩니다.",
                ][family]
            )
            rng.shuffle(facts)
            answer = absent
            question = [
                f"{obj}을 가져오지 않은 사람은?",
                f"누가 {obj}을 안 가져왔나요?",
                f"{obj}을 가져온 적 없는 사람은?",
                f"{obj} 지참 사실이 부정되는 인물을 답하세요.",
                f"두 인물 가운데 {obj}이 없는 사람을 고르세요.",
                f"어느 인물에 대해 {obj}을 가져왔다는 판단을 내릴 수 없습니까?",
                f"위 사실에 따라 {obj} 지참자로 분류되지 않는 사람은 누구입니까?",
            ][family]
        else:
            quantity = rng.randint(0, 99)
            facts = [f"처음 {first}가 가진 {obj}의 수: {quantity}개."]
            for _ in range(event_count):
                add = bool(rng.getrandbits(1))
                delta = rng.randint(0, 49 if add else quantity)
                order.append(delta if add else -delta)
                quantity += delta if add else -delta
                operation = "추가" if add else "제거"
                facts.append(
                    [
                        f"그 뒤 {obj} {delta}개를 {'받았습니다' if add else '주었습니다'}.",
                        f"이어서 {obj}의 수가 {delta}개 {'늘었습니다' if add else '줄었습니다'}.",
                        f"다음 변화는 {obj} {delta}개의 {operation}입니다.",
                        f"후속 사건에서 {obj} {delta}개가 {'그에게 더해졌습니다' if add else '그에게서 없어졌습니다'}.",
                        f"뒤이은 수량 조정은 {obj} {delta}개 {'증가' if add else '감소'}입니다.",
                        f"다음 재고 기록에서 {obj} {delta}개가 {'유입' if add else '유출'}됐습니다.",
                        f"이후 보유량을 {delta}개 {'더 크게' if add else '더 작게'} 만드는 사건이 발생했습니다.",
                    ][family]
                )
            answer = str(quantity)
            question = [
                f"지금 {first}의 {obj}은 몇 개인가요?",
                f"남은 {obj}의 개수는?",
                "마지막 수량을 숫자로 쓰세요.",
                "변화를 모두 반영한 최종 보유량을 답하세요.",
                "기록 종료 시의 재고량을 계산하세요.",
                "모든 유입과 유출 이후 남아 있는 수량을 정수로 적어 주세요.",
                "가장 마지막 사건까지 처리한 보유 개수를 계산해 주세요.",
            ][family]
        cue = {"train": rng.choice(["답: ", "정답은: ", "대답: "]), "val": "결론: ", "test": "최종 답변: "}[forms]
        separator = rng.choice([" ", "\n"]) if forms == "train" else "\n"
        prompt = (
            rng.choice(PREFIXES[forms])
            + "사건은 시간 순서대로 적혀 있습니다. "
            + separator.join(facts)
            + "\n"
            + question
            + "\n"
            + cue
        )
        yield {
            "text": prompt + answer,
            "prompt": prompt,
            "answer": answer,
            "task": task,
            "split": split,
            "source": "program_rules",
            "render_family": family,
            "event_order": order,
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}],
        }
