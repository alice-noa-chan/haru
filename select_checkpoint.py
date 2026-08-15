"""Report which checkpoint the training log actually favours, and why.

`best.pt` is saved against a selection score that adds the relation loss to the
language-model loss:

    selection = final_loss + COUNTERFACTUAL_LOSS_WEIGHT * relation_loss

That is the right objective when both terms are learning. It stops being the
right objective when one term saturates on its training distribution and its
held-out value turns to noise, because then the sum is dominated by the noisy
term and `best.pt` tracks that noise instead of model quality.

This reads the run's train_log.jsonl and reports the best checkpoint under each
criterion, along with how much the two disagree. It does not modify anything.
Use it before exporting, to decide whether best.pt or latest.pt is the honest
choice for a release.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import config


def validation_records(path: Path) -> list[dict]:
    """Every validation entry, flattened to the fields worth comparing."""

    records = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "validation" not in row:
            continue

        relation = row.get("counterfactual_validation", {})
        records.append(
            {
                "step": row.get("step", 0),
                "tokens": row.get("tokens_seen", 0),
                "lm_loss": row["validation"].get("final_loss"),
                "relation_loss": relation.get("loss"),
                "relation_pairs": relation.get("strict_pair_accuracy"),
                "selection": row.get("selection_loss"),
            }
        )
    return [r for r in records if r["lm_loss"] is not None]


def spread(values: list[float]) -> float:
    return max(values) - min(values) if values else 0.0


def report(records: list[dict]) -> None:
    by_lm = min(records, key=lambda r: r["lm_loss"])
    scored = [r for r in records if r["selection"] is not None]
    by_selection = min(scored, key=lambda r: r["selection"]) if scored else None

    print(f"{len(records)} validation points, steps {records[0]['step']:,} to {records[-1]['step']:,}\n")

    print(f"{'criterion':<26}{'step':>9}{'LM loss':>10}{'relation':>10}{'pairs':>8}{'selection':>11}")
    print("-" * 74)
    print(
        f"{'language-model loss':<26}{by_lm['step']:>9,}{by_lm['lm_loss']:>10.4f}"
        f"{by_lm['relation_loss'] or 0:>10.4f}{by_lm['relation_pairs'] or 0:>8.0%}"
        f"{by_lm['selection'] or 0:>11.4f}"
    )
    if by_selection is not None:
        print(
            f"{'selection score (best.pt)':<26}{by_selection['step']:>9,}{by_selection['lm_loss']:>10.4f}"
            f"{by_selection['relation_loss'] or 0:>10.4f}{by_selection['relation_pairs'] or 0:>8.0%}"
            f"{by_selection['selection']:>11.4f}"
        )

    # The comparison that matters: how much variation each term contributes over
    # the recent past. If the relation term moves far more than the language
    # model term, the sum is choosing on the relation term.
    tail = records[-20:]
    lm_spread = spread([r["lm_loss"] for r in tail])
    relation_spread = spread([r["relation_loss"] for r in tail if r["relation_loss"] is not None])
    weighted = config.COUNTERFACTUAL_LOSS_WEIGHT * relation_spread

    print(f"\nover the last {len(tail)} validations:")
    print(f"  language-model loss varies by   {lm_spread:.4f}")
    print(
        f"  relation loss varies by         {relation_spread:.4f}  (x{config.COUNTERFACTUAL_LOSS_WEIGHT} = {weighted:.4f})"
    )

    if weighted > lm_spread:
        ratio = weighted / lm_spread if lm_spread else float("inf")
        print(
            f"\n  The relation term drives the selection score, {ratio:.0f}x more than the language\n"
            f"  model does. best.pt is being chosen by that term. Prefer the checkpoint at\n"
            f"  step {by_lm['step']:,} on language-model loss, or latest.pt if it is the newest."
        )
    else:
        print("\n  The language-model term dominates the selection score; best.pt is a sound choice.")

    # Held-out relation accuracy is the honest read on whether that objective
    # generalised at all, separately from which checkpoint to ship.
    pairs = [r["relation_pairs"] for r in records if r["relation_pairs"] is not None]
    if pairs:
        early = sum(pairs[:5]) / len(pairs[:5])
        late = sum(pairs[-5:]) / len(pairs[-5:])
        print(f"\nheld-out relation pairs: {early:.0%} early -> {late:.0%} late", end="")
        print("  (not generalising)" if late <= early else "  (improving)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", type=Path, default=config.TRAIN_LOG_PATH)
    args = parser.parse_args()

    if not args.log.exists():
        raise FileNotFoundError(f"No training log at {args.log}")

    records = validation_records(args.log)
    if not records:
        raise ValueError(f"{args.log} holds no validation records yet")
    report(records)


if __name__ == "__main__":
    main()
