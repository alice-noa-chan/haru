# Results

Machine-readable results for every measurement RESEARCH.md and the project
README quote. `runs/` is excluded from Git because it holds checkpoints and
packed data, which left the documented figures with no checked-in source.

Each file records the seeds, step count, batch size, relation weight, tokenizer,
and full model configuration of every arm, plus per-seed validation losses,
paired deltas, strict pair accuracies, parameter counts, and measured FLOPs.

| File | Produced by | Reported in |
|---|---|---|
| `architecture_comparison_3seed.json` | `compare_architectures.py --scale small --steps 400 --seeds 3` | RESEARCH.md, "Direction test" |
| `architecture_ablation_3seed.json` | `compare_architectures.py --scale small --steps 300 --seeds 3 --ablate` | RESEARCH.md, "The result: row two fired"; README, "Result at compact scale" |

Both ran on CPU with the 8,192-entry tokenizer at reduced model scale. They are
direction tests, not release-scale evidence. Step counts differ between the two
files, so figures may be compared within a file but not across them.

Regenerating either command overwrites `runs/`, not this directory. Copy a new
result here deliberately, alongside the documentation change that cites it.
