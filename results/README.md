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
| `architecture_comparison_release.json` | `compare_architectures.py --scale release` (6,000 steps, 3 seeds) | RESEARCH.md, "Release-scale result"; README, "Result at release scale" |
| `architecture_release_ablate_deepsup_6000steps_3seeds.json` | `compare_architectures.py --scale release --ablate --deep-supervision-arms` (6,000 steps, 3 seeds) | RESEARCH.md, "Eight-arm release table" |
| `architecture_release_ablate_combine_6000steps_3seeds.json` | `compare_architectures.py --scale release --ablate --combine` (6,000 steps, 3 seeds) | RESEARCH.md, "Does the combination compose" |
| `korean_benchmarks_v1.1.json` | `evaluate_korean.py runs/haru-v2-binding/transformers` | README, "Korean benchmarks" |
| `korean_benchmarks_v2.0_midtraining.json` | `evaluate_korean.py runs/haru-v2-unfolded/transformers` at 39.6% of the v2.0 budget | README, "Korean benchmarks"; a mid-training probe, not a release score |
| `korean_benchmarks_v2.0.json` | `evaluate_korean.py runs/haru-v2-unfolded/transformers` at the released step 32,500 checkpoint | README, "Korean benchmarks" |
| `korean_benchmarks_v2.0_step31000.json` | the same, at step 31,000; the second of two points used to decide whether more tokens were still buying benchmark accuracy | README, "Korean benchmarks" |
| `korean_benchmarks_minpeter_tiny-ko-20m-base.json` | same harness and task list, run locally on that model | README, "Against other sub-20M Korean models" |
| `korean_benchmarks_minpeter_tiny-ko-20m-sft.json` | same harness and task list, run locally on that model | README, "Against other sub-20M Korean models" |
| `korean_benchmarks_gaon12_haru_1.1.json` | v1.1 re-scored locally through the same harness as the comparison models; reproduces the published v1.1 figures to within 0.0002 per task | README, "Against other sub-20M Korean models" |
| `decontamination_textbooks.json` | `decontaminate.py data/textbooks.txt` | data/datasets.md, decontamination log |

The first two ran on CPU with the 8,192-entry tokenizer at reduced model scale
and are direction tests. The two release files ran on GPUs at the full
architecture with the 12,000-entry tokenizer.

The eight-arm file supersedes the three-arm one. Both were run at the same
budget, but only the eight-arm table includes the deep-supervision controls,
and it does not reproduce the three-arm table's compute-matched result.

Step counts, model scale, and tokenizer differ between the architecture files,
so figures may be compared within a file but never across them.

The benchmark and decontamination files are not architecture comparisons. The
first records where a released checkpoint stands against outside models; the
second records how much text a corpus lost to benchmark overlap, which must be
checked before any benchmark score from that corpus is believed.

Regenerating either command overwrites `runs/`, not this directory. Copy a new
result here deliberately, alongside the documentation change that cites it.
