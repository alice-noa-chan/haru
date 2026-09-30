# Train Haru on your GPU

Haru's dense decoder targets Korean children's-story continuation. The student
candidates are all below 18 million parameters. The pipeline compares three
architectures, grows the winner into a larger teacher, trains that teacher,
and distills separate base and chat students. It uses no external LLM weights
or generated teacher data.

## Install

Use Python 3.10 or newer. Install a PyTorch build compatible with your GPU and
CUDA driver using the [official PyTorch selector](https://pytorch.org/get-started/locally/),
then install the model's other dependencies:

```bash
python -m pip install -r requirements-train.txt
```

`liger-kernel` is optional on supported Linux/CUDA systems. The pipeline checks
loss and gradient agreement before selecting an optimized kernel. It uses a
verified PyTorch fallback when a kernel is missing or fails that check.

## Prepare your data

Put one document per line in each of these UTF-8 files under `data/`:

| File | Domain |
| --- | --- |
| `textbooks.clean.txt` | instructional text |
| `webtext.clean.txt` | general web text |
| `wikipedia.clean.txt` | encyclopedic text |
| `data.txt` | Korean children's stories |

Use data you have permission to train on. Training-source records are kept locally.
The files themselves are excluded from Git. The preparation step normalizes and
deduplicates documents, creates disjoint train/validation/test splits, learns a
12,000-token BPE including chat role tokens, and packs the streams:

```bash
python -m haru.data --source-root data --output packed/haru-v3
python -m haru.story_view --base packed/haru-v3 --output packed/haru-v3-story
python -m haru.expand_rules --base packed/haru-v3 --output packed/haru-v3-rules-1m
```

The story view shares the tokenizer and document splits with the prepared base
data. Its base-training token mixture is 50% stories, 25% instructional text,
15% web text, 5% encyclopedic text, and 5% programmatically answered rules.
The expanded rules are a distinct data version used from the teacher stage.
The rules and chat answers come from deterministic programs or source text.

If data is prepared on another machine, copy the `packed/` directories to the
GPU machine. `python -m haru.archive bundle SOURCE OUTPUT.tar.zst --training-data`
creates a hashed archive; `python -m haru.archive unpack ARCHIVE DESTINATION`
checks the archive and every extracted file. An ordinary filesystem copy works
as well.

## Train and resume

From the repository root, run:

```bash
python train.py --data packed/haru-v3-story \
  --teacher-rules-overlay packed/haru-v3-rules-1m \
  --output runs/haru-v3 --device cuda
```

The same command resumes from saved checkpoints. Checkpoints include weights,
optimizer and schedule state, RNG state, sampler position, data/tokenizer hashes,
and the source commit. They are written atomically at optimizer-step boundaries.
`--max-seconds` is available for a time-limited session; restart with the same
data, tokenizer, architecture, and output directory to continue. Changing the
GPU preserves training state but does not promise bitwise-identical operations.

The effective batch is 131,072 tokens through gradient accumulation. The
pipeline measures supported attention/loss kernels and microbatch sizes on the
current GPU, verifies their output and gradients, then selects the fastest
correct setting. It chooses candidate learning rates on held-out story BPC,
replicates the top two candidates with another seed, and does not use test or
public-benchmark data for selection. Defaults are 2B teacher-base tokens,
500M student-base tokens, and 10M tokens for each chat stage; these are upper
limits, not mandatory epochs. The best validation checkpoints are kept.

For one training phase or custom schedules, see `python -m haru.training --help`.
The older CFRD training command remains `python train_legacy.py`.

## Export and evaluate

The four output phases are `teacher-base`, `student-base`, `teacher-chat`, and
`student-chat`. Export a completed checkpoint locally before publishing:

```bash
python -m haru.export \
  --checkpoint runs/haru-v3/student-chat/best.pt \
  --tokenizer packed/haru-v3-story/tokenizer.model \
  --output runs/haru-v3/exports/student-chat
```

The export verifies independent Transformers loading, writes FP32 Safetensors,
and retains a trusted-source resume checkpoint. `AutoTokenizer`,
`AutoModelForCausalLM`, `generate`, `apply_chat_template`, and incremental
`use_cache=True` generation are supported. Publishing is an explicit optional
step via `--repo-id YOUR_ORG/YOUR_MODEL`.

For checkpoints trained before the public source layout changed, pass
`--public-source-commit FULL_GIT_SHA` to link the corresponding public source
revision. The export metadata keeps the checkpoint's original training commit
separately for provenance.

After model selection and training are complete, run the final-only comparison:

```bash
python -m haru.evaluate \
  --model runs/haru-v2-unfolded/transformers \
  --model runs/haru-v3/exports/student-base \
  --model runs/haru-v3/exports/student-chat \
  --data packed/haru-v3-story \
  --output results/haru_dense_final.json
```

This records raw-text BPC on the same documents at context 512, native context
1024 for the dense model, story generations, relation generalization, and CPU
speed. Compare generated stories for character, object, and event continuity;
a lower BPC alone does not establish better storytelling. Run public benchmarks
only after this selection is frozen. Do not describe the model as the best under
18M parameters without comparable evidence.
