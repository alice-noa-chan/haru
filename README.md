# Haru

<p align="center">
  <img src="assets/haru.png" alt="Haru character" width="420">
</p>

<p align="center"><strong>Haru</strong> — the character representing Haru in the model family.</p>

Haru is a compact Korean story continuation language model. Its custom Causal
Folded Recurrent Decoder (CFRD) uses local causal attention, compressed summary
memory, and reusable decoder cells. Measurements at release scale show the cell
reuse is the one part that does not pay for itself; see
[Result at release scale](#result-at-release-scale). Haru v1.1 is the current 11.6-million-
parameter release; the original 6.8-million-parameter checkpoint remains
available as Haru v1.0. Both support recurrent inference depths 2, 4, and 6.

- GitHub: [alice-noa-chan/haru](https://github.com/alice-noa-chan/haru)
- Model v1.1: [gaon12/haru_1.1](https://huggingface.co/gaon12/haru_1.1)
- Model v1.0: [gaon12/haru](https://huggingface.co/gaon12/haru)
- Demo: [gaon12/haru](https://huggingface.co/spaces/gaon12/haru)

Haru is a research prototype for story continuation. It is not an instruction
model, a factual assistant, or a safety-reviewed product.

## Releases

| Version | GitHub | Hugging Face | Status |
|---|---|---|---|
| 1.1 | [`v1.1.0`](https://github.com/alice-noa-chan/haru/releases/tag/v1.1.0) | [`gaon12/haru_1.1`](https://huggingface.co/gaon12/haru_1.1) | Current |
| 1.0 | [`v1.0.0`](https://github.com/alice-noa-chan/haru/releases/tag/v1.0.0) | [`gaon12/haru`](https://huggingface.co/gaon12/haru) | Legacy |

## Haru v1.1 at a glance

| Setting | Value |
|---|---:|
| Vocabulary | 12,000 |
| Context length | 512 tokens |
| Local chunk | 64 tokens |
| Hidden size | 384 |
| Query / KV heads | 6 / 2 |
| Physical decoder cells | 3 |
| Recurrent depth | 6 |
| FFN size | 1,024 |
| Full-context binding block | enabled |
| Trainable parameters | 11,634,459 |
| Best-checkpoint training tokens | 753,664,000 |

The three physical cells run in this order:

```text
A -> B -> C -> A -> B -> C
```

The shared language-model head supervises depths 2, 4, and 6. Depth 4 is a
useful CPU setting; depth 6 gives the best measured validation quality.

## What changed in v1.1

Version 1.1 uses a separate `haru-v2-binding` run and a separate 12,000-token
tokenizer. It does not overwrite the v1.0 artifacts.

| Setting | Value |
|---|---:|
| Vocabulary | 12,000 |
| Physical decoder cells | 3 |
| Recurrent depth | 6 |
| Full-context binding block | enabled |
| Trainable parameters | 11,634,459 |

The extra cell reduces forced parameter sharing. The final binding block gives
all tokens one independent causal-attention pass after the recurrent stack,
so entity, role, and attribute evidence need not survive only through local
chunks and compressed summaries.

Training also adds a 0.20-weight auxiliary relation objective. Each step draws
fresh entity permutations across location, state, ownership, transfer, and
speaker-attribution pairs. The loss compares only candidate-answer token spans
and requires the preferred answer to flip in both counterfactual directions.
Held-out names and phrasings are evaluated separately; best-checkpoint
selection combines language-model loss with this relation loss.

## Install

Haru requires Python 3.11 or newer.

```bash
python -m pip install -r requirements.txt
```

## Load with Transformers

The public inference format uses Safetensors and standard Transformers
AutoClasses. It does not load the training `.pt` checkpoint.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "gaon12/haru_1.1"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
).eval()

inputs = tokenizer("작은 마을에 조용한 아침이 찾아왔어요.", return_tensors="pt")
output = model.generate(
    **inputs,
    max_new_tokens=120,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    top_k=40,
    repetition_penalty=1.08,
    use_cache=False,
)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

`trust_remote_code=True` is required because CFRD is a custom Transformers
architecture. Review and pin a repository revision before using remote custom
code in a production environment.

## Generate from a local export

Set the prompt and sampling values in `config.py`, then run:

```bash
python generate.py
```

`generate.py` loads `AutoTokenizer` and `AutoModelForCausalLM`. Set
`INFERENCE_RECURSIONS` to 2, 4, or 6 in `config.py`.

## Architecture

Every recurrent step has three paths:

1. Local attention reads earlier tokens inside the same 64-token chunk.
2. Summary memory pools completed chunks into four compact slots and exposes
   only earlier chunks.
3. A SwiGLU feed-forward network transforms each token independently.

Summary memory has a learned per-head recency bias. Korean token embeddings
also receive a small projection of onset, vowel, coda, word-boundary, digit,
ASCII, punctuation, byte-fallback, and token-length features.

The v1.1 configuration then applies a non-recurrent full-context
causal attention and SwiGLU block before the shared language-model head. Older
configuration files do not enable this block, so existing exports remain
loadable with their original architecture.

See [RESEARCH.md](RESEARCH.md) for related work.

## Version comparison

All three models were evaluated with their native tokenizers. Cross-model
forced-choice accuracy uses the same 42 cases for which both candidates have
matched token lengths in every tokenizer. Bits per character (BPC) normalizes
the shared-text language-model score for tokenizer differences; lower is
better.

| Model | Parameters | Factual choice | Counterfactual strict pairs | BPC |
|---|---:|---:|---:|---:|
| Haru v1.0 | 6,793,363 | 18/42 (42.9%) | 0/100 | 2.668 |
| Haru v1.1 | 11,634,459 | **20/42 (47.6%)** | **21/100** | **2.631** |
| Tiny-Ko-Stories-35M | 34,217,856 | 15/42 (35.7%) | 2/100 | 2.681 |

The v1.1 counterfactual gain is concentrated in location (16/20 strict pairs)
and ownership (4/20). Speaker attribution and transfer remain 0/20, so v1.1 is
still a research checkpoint rather than a solved entity-binding model. Run
`python compare_models.py` to reproduce the three-way comparison.

## Haru v1.1 evaluation

The final held-out evaluation used 200 fixed batches at each supervised depth.

| Inference recurrences | Validation loss | Perplexity |
|---:|---:|---:|
| 2 | 2.95689 | 19.238 |
| 4 | 2.12258 | 8.353 |
| 6 | 1.92052 | 6.825 |

Per-token perplexity is comparable only for models using the same tokenizer.
For different tokenizers, use byte- or character-normalized negative
log-likelihood and report actual latency or FLOPs.

## Architecture comparison

CFRD reuses three physical cells across six recurrences, so it does not spend
compute the way its parameter count suggests. Measured with
`torch.utils.flop_counter` on a 512-token forward:

| Model | Parameters | GFLOPs | Blocks applied |
|---|---:|---:|---:|
| Haru v1.1 CFRD (depth 6) | 11,634,459 | 17.31 | 7 |
| Dense 4 layers x FFN 1184 | 11,669,377 | 11.95 | 4 |
| Dense 7 layers x FFN 1184 | 16,943,233 | 17.34 | 7 |

Holding parameters fixed leaves CFRD with about 1.45x the forward FLOPs, so a
lone "same parameter count" comparison is not a fair test. `compare_architectures.py`
therefore trains two controls on identical windows in an identical order:

- **parameter-matched** (+0.30% parameters, -30.98% FLOPs). CFRD must win this
  for recurrent folding to be worth anything.
- **compute-matched** (+45.63% parameters, +0.21% FLOPs). Only this arm can
  support a claim about quality per parameter.

```bash
python compare_architectures.py --scale small              # CPU direction check
python compare_architectures.py --scale small --ablate     # plus within-CFRD arms
python compare_architectures.py --scale release            # GPU, config.py architecture
```

On the cloud, the release-scale table is one command against the corpus and
tokenizer already on the volume:

```bash
python -m the cloud run --detach cloud_train.py --action compare
python -m the cloud run --detach cloud_train.py --action compare --ablate
```

Baseline shapes are derived from the CFRD configuration rather than hard-coded,
so they stay matched when the architecture changes. To train a single dense arm
through the normal pipeline instead, set `MODEL_ARCH = "dense-baseline"` and a
new `RUN_NAME` in `config.py`.

### Result at release scale

Three seeds, 6,000 steps, 49,152,000 tokens per arm per seed, at the release
architecture on an a GPU, with the relation objective at weight 0.20 in every
arm. A positive delta means CFRD won. "Beyond spread" marks deltas larger than
the spread of the paired per-seed differences; the others are noise.

| Arm | Parameters | GFLOPs | Validation loss | vs CFRD | Beyond spread | Strict pairs |
|---|---:|---:|---:|---:|:---:|---:|
| CFRD without cell sharing | 17,047,725 | 17.31 | **3.3212** | -0.0586 | yes | 0.370 |
| CFRD with full-context cells | 10,944,393 | 16.03 | 3.3565 | -0.0233 | yes | 0.253 |
| CFRD without deep supervision | 11,634,459 | 17.31 | 3.3744 | -0.0054 | yes | **0.377** |
| CFRD | 11,634,459 | 17.31 | 3.3798 | reference | | 0.263 |
| Dense, parameter-matched, deep-supervised | 11,669,377 | 11.95 | 3.3842 | +0.0044 | no | 0.313 |
| Dense, compute-matched | 16,943,233 | 17.34 | 3.3846 | +0.0048 | no | 0.300 |
| Dense, parameter-matched | 11,669,377 | 11.95 | 3.4294 | +0.0496 | yes | 0.263 |
| CFRD without binding block | 10,060,057 | 15.70 | 3.4664 | +0.0866 | yes | 0.267 |

**CFRD beats a plain dense decoder, but not because of the folding.** CFRD
trains with auxiliary exits at depths 2 and 4 that a plain dense baseline does
not have. Giving the dense arm the same auxiliary exits costs no parameters,
because the head is tied to the embedding, and closes 91% of the gap. What
remains, 0.0044, is inside the noise.

**The largest effect in the table is removing parameter sharing.** Giving each
recurrence its own cell wins by 0.0586 on every seed, and also scores best on
entity binding. The fold is the one component that does not pay for itself.
The rest do: removing the binding block is the worst result measured, and at a
matched budget the unfolded model still beats the compute-matched dense arm by
0.0634.

Strict pair accuracy has a chance level of 0.25 and is too noisy at this budget
to rank arms: CFRD's own three seeds ran 0.10, 0.23, and 0.46. Only the
unfolded and no-deep-supervision arms clear chance on every seed. See
[RESEARCH.md](RESEARCH.md) for per-seed figures and what the table rules out.

### Result at compact scale

At 3.3M parameters and a far shorter budget the same table comes out the other
way, which is why the release-scale run above was needed.

A CPU-scale replication: six arms, three seeds, 300 steps,
614,400 tokens per arm per seed, at 3.3M parameters rather than 11.6M. Every
paired delta exceeded its own spread across seeds.

| Arm | Parameters | GFLOPs | Validation loss | vs CFRD | Strict pairs |
|---|---:|---:|---:|---:|---:|
| CFRD | 3,339,673 | 1.93 | 4.7273 | reference | 0.110 |
| Dense, parameter-matched | 3,364,801 | 1.72 | **4.6557** | -0.0715 | 0.203 |
| Dense, compute-matched | 3,809,089 | 1.95 | **4.6518** | -0.0754 | **0.343** |
| CFRD without binding block | 2,945,687 | 1.73 | 4.7920 | +0.0647 | 0.043 |
| CFRD without cell sharing | 3,791,327 | 1.93 | 4.6958 | -0.0315 | 0.043 |
| CFRD with full-context cells | 3,166,665 | 1.82 | 4.7911 | +0.0638 | 0.050 |

No CFRD configuration reached a plain dense decoder on either metric, at
matched parameters or at matched compute. Strict pair accuracy has a chance
level of 0.25, since both directions of a pair must flip.

Both tables are accurate where they were measured. The release-scale run has
roughly 80x the tokens per arm and 3.5x the parameters, and that difference is
what separates the two outcomes, so the compact-scale result should be read as
a lower bound on where folding starts to pay rather than as a verdict on the
architecture. See [RESEARCH.md](RESEARCH.md) for per-seed figures.

## Train from source

### 1. Add text

Place `.txt` or `.jsonl` files under `data/`. TXT uses one sample per line.
JSONL uses the `text` field. Source files under `data/` are excluded from Git.

### 2. Train the tokenizer

```bash
python tokenizer_train.py
```

### 3. Pack train and validation streams

```bash
python prepare_data.py
```

Text is normalized to Unicode NFC. A stable BLAKE2 hash assigns each record to
one split. BOS and EOS tokens separate records in the packed stream.

### 4. Run tests

```bash
python smoke_test.py
```

The suite checks shapes, gradients, causality, partial chunks, exact checkpoint
resume (including the dynamic relation sampler), counterfactual-pair leakage
and gradients, Safetensors parity, AutoClass loading, tokenizer round trips,
and the default parameter count.

### 5. Train and evaluate

```bash
python train.py
python evaluate.py
```

Review model, optimizer, schedule, and path settings in `config.py` first. Use
a new `RUN_NAME` when changing an architecture or training schedule. The
current defaults already isolate the new tokenizer, packed data, checkpoints,
and logs under `haru-v2-binding` paths.

### 6. Export

```bash
python export_transformers.py
```

The exporter writes the tokenizer, custom Transformers code, configuration,
model card, generation configuration, and `model.safetensors` under
`runs/<RUN_NAME>/transformers/`. It then reloads the directory through
AutoClasses and verifies all serialized tensors and output logits.

## Train on the cloud

Install and authenticate the cloud:

```bash
python -m pip install -r requirements-the cloud.txt
python -m the cloud setup
```

Prepare data and start a detached a GPU run:

```bash
python -m the cloud run cloud_train.py --action benchmark
python -m the cloud run cloud_train.py --action upload
python -m the cloud run cloud_train.py --action prepare
python -m the cloud run --detach cloud_train.py --action train --gpu a GPU --batch-size 16
```

Evaluate and export after training:

```bash
python -m the cloud run cloud_train.py --action evaluate
python -m the cloud run cloud_train.py --action export
```

Compare CFRD against matched dense baselines at the release architecture:

```bash
python -m the cloud run --detach cloud_train.py --action compare
python -m the cloud run --detach cloud_train.py --action compare --ablate
```

Defaults are 6,000 steps and 3 seeds, or 49.2M tokens per arm per seed.
`--action benchmark` reports measured throughput and a cost estimate for 800M
tokens; the three-arm table costs roughly 0.55x that figure and the six-arm
`--ablate` form roughly 1.1x.

The the cloud app and persistent Volume are named `haru` and `haru-training`.
a GPU uses fp16 with gradient scaling; a GPU uses bf16.

## Compare on the cloud Cloud

`cloud_train.py` runs the same comparison on the cloud, for when the cloud credits run
out. It pins identical package versions, so a result cannot differ because of
where it ran.

```bash
uv tool install the cloud-client
the cloud configure default --token <API_KEY>
the cloud volume create haru-training
```

Upload the corpus and tokenizer once. **Do this from Linux or WSL, not from
Windows**: the Windows client joins the volume name and key with a backslash,
and `the cloud cp --no-multipart` reports a completed transfer while writing
nothing at all. Always confirm afterwards.

```bash
the cloud cp data/data.txt the cloud://haru-training/data/data.txt
the cloud cp tokenizer/tiny_ko_12k.model the cloud://haru-training/tokenizer/tiny_ko_12k.model
the cloud ls haru-training/data          # must show data.txt, not "0 items"
```

Then run any arm set:

```bash
python cloud_train.py                                    # 3 arms
python cloud_train.py --ablate                           # + 3 within-CFRD arms
python cloud_train.py --ablate --deep-supervision-arms   # + 2 supervision arms
```

the cloud has no a GPU, so the function requests an a GPU. Every arm shares the same
GPU, so the comparison is unaffected. `headless=True` keeps the run alive after
the client disconnects, which matters because these tables run for hours.

Results are named after the arm set and budget, for example
`architecture_release_ablate_deepsup_6000steps_3seeds.json`, and land on the
volume under `runs/<RUN_NAME>/`. Fetch one with:

```bash
the cloud cp the cloud://haru-training/runs/haru-v2-binding/<name>.json results/
```

## Project files

| File | Purpose |
|---|---|
| `config.py` | Paths, architecture, training, and generation settings |
| `model.py` | CFRD core model |
| `configuration_cfrd.py` | Transformers configuration |
| `modeling_cfrd.py` | Transformers causal-LM implementation |
| `tokenization_cfrd.py` | Transformers tokenizer implementation |
| `tokenizer_train.py` | SentencePiece tokenizer training |
| `prepare_data.py` | Packed token-stream creation |
| `train.py` | Training, validation, checkpoints, and exact resume |
| `evaluate.py` | Recurrent-depth evaluation |
| `export_transformers.py` | Safetensors and AutoClass export |
| `baseline_model.py` | Dense Transformer control for CFRD ablations |
| `model_factory.py` | Architecture selection for training and evaluation |
| `compare_architectures.py` | CFRD versus matched dense baselines |
| `compare_models.py` | Reproducible v1.0/v1.1/35M comparison |
| `generate.py` | Transformers-based local generation |
| `cloud_train.py` | the cloud benchmark, preparation, training, and export |
| `cloud_train.py` | the cloud Cloud architecture comparison |
| `smoke_test.py` | Fast correctness tests |
| `NOTICE.md` | Required training-data attribution |
| `RESEARCH.md` | Related architecture research |

## Known limitations

- Fixed chunk boundaries do not follow sentences or story boundaries.
- Summary memory can lose names, quotations, and exact event details.
- Generation has no KV or recurrent-state cache yet.
- Longer continuations may repeat ideas or drift between entities.
- Speaker attribution and transfer bindings remain weak in v1.1.
- No released Haru checkpoint has itself been compared against a matched dense
  Transformer. The release-scale comparison trains fresh arms at 49.2M tokens
  each, so it does not retroactively validate the v1.0 or v1.1 artifacts.
- CFRD's advantage over a parameter-matched dense decoder is explained by its
  auxiliary exits, not by the fold. With supervision equalized the two are
  indistinguishable, and removing parameter sharing is the largest measured
  gain in the table.
- The shared 4e-4 learning rate is CFRD's own tuned value and remains the last
  uncontrolled variable large enough to affect the ranking.

## Training data notice

The training corpus is not included in this repository or model release. The
minimal attribution required by its license is recorded in [NOTICE.md](NOTICE.md).

## License

Haru source code and model weights are released under the
[MIT License](LICENSE). The training dataset remains subject to its separate
CC BY 4.0 license described in [NOTICE.md](NOTICE.md).
