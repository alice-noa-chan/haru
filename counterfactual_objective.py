from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import torch
import torch.nn.functional as F

from counterfactual_data import CounterfactualPair


class TextEncoder(Protocol):
    pad_id: int

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]: ...


@dataclass(slots=True)
class CounterfactualBatch:
    input_ids: torch.Tensor
    target_ids: torch.Tensor
    score_mask: torch.Tensor
    pair_count: int


@dataclass(slots=True)
class CounterfactualResult:
    loss: torch.Tensor
    decision_accuracy: torch.Tensor
    strict_pair_accuracy: torch.Tensor
    mean_margin: torch.Tensor


def encode_counterfactual_pairs(
    pairs: Sequence[CounterfactualPair],
    tokenizer: TextEncoder,
    context_length: int,
    device: torch.device,
) -> CounterfactualBatch:
    """Encode four candidate continuations per pair and mark answer tokens only."""

    if not pairs:
        raise ValueError("At least one counterfactual pair is required")

    rows: list[tuple[list[int], int]] = []
    for pair in pairs:
        for prompt in (pair.prompt_a, pair.prompt_b):
            prefix_ids = tokenizer.encode(prompt.rstrip() + "\n", add_bos=True)
            for candidate in pair.candidates:
                full_ids = tokenizer.encode(prompt.rstrip() + "\n" + candidate, add_bos=True)
                if full_ids[: len(prefix_ids)] != prefix_ids:
                    raise ValueError("Tokenizer changed the prompt prefix after appending a candidate")
                if len(full_ids) <= len(prefix_ids):
                    raise ValueError("A candidate must contain at least one token")
                if len(full_ids) - 1 > context_length:
                    raise ValueError(
                        f"Counterfactual sequence has {len(full_ids) - 1} tokens, exceeding context {context_length}"
                    )
                rows.append((full_ids, len(prefix_ids)))

    max_time = max(len(full_ids) - 1 for full_ids, _ in rows)
    input_ids = torch.full((len(rows), max_time), tokenizer.pad_id, dtype=torch.long, device=device)
    target_ids = torch.full_like(input_ids, tokenizer.pad_id)
    score_mask = torch.zeros_like(input_ids, dtype=torch.bool)

    for row_index, (full_ids, candidate_start) in enumerate(rows):
        input_row = torch.tensor(full_ids[:-1], dtype=torch.long, device=device)
        target_row = torch.tensor(full_ids[1:], dtype=torch.long, device=device)
        time = input_row.numel()
        input_ids[row_index, :time] = input_row
        target_ids[row_index, :time] = target_row
        score_mask[row_index, candidate_start - 1 : time] = True

    return CounterfactualBatch(
        input_ids=input_ids,
        target_ids=target_ids,
        score_mask=score_mask,
        pair_count=len(pairs),
    )


def candidate_scores(logits: torch.Tensor, batch: CounterfactualBatch) -> torch.Tensor:
    """Return mean answer-token log probabilities shaped [pair, variant, candidate]."""

    if logits.shape[:2] != batch.input_ids.shape:
        raise ValueError("Logit and counterfactual batch dimensions differ")

    token_log_probs = F.log_softmax(logits.float(), dim=-1)
    selected = token_log_probs.gather(-1, batch.target_ids.unsqueeze(-1)).squeeze(-1)
    mask = batch.score_mask.to(dtype=selected.dtype)
    row_scores = (selected * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)
    return row_scores.view(batch.pair_count, 2, 2)


def counterfactual_ranking_result(
    model: torch.nn.Module,
    batch: CounterfactualBatch,
    margin: float,
) -> CounterfactualResult:
    """Require the preferred candidate to flip across both prompt variants."""

    if margin < 0.0:
        raise ValueError("margin must be non-negative")

    scores = candidate_scores(model(batch.input_ids).logits, batch)
    first_margin = scores[:, 0, 0] - scores[:, 0, 1]
    second_margin = scores[:, 1, 1] - scores[:, 1, 0]
    margins = torch.stack((first_margin, second_margin), dim=1)

    # Softplus keeps useful gradients after crossing the target margin while
    # strongly penalizing identity shortcuts that fail one side of the pair.
    loss = F.softplus(margin - margins).mean()
    correct = margins > 0.0
    return CounterfactualResult(
        loss=loss,
        decision_accuracy=correct.float().mean(),
        strict_pair_accuracy=correct.all(dim=1).float().mean(),
        mean_margin=margins.mean(),
    )
