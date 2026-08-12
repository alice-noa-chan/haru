# Haru

<p align="center">
  <img src="assets/haru.png" alt="Haru character" width="420">
</p>

<p align="center"><strong>Haru</strong> — the character representing Haru in the model family.</p>

Haru is a compact Korean story continuation language model. Its custom Causal
Folded Recurrent Decoder (CFRD) uses local causal attention, compressed summary
memory, and reusable decoder cells. The released checkpoint has 6.8 million
parameters and can trade inference cost for quality with recurrent depths 2,
4, and 6. The repository's next-training defaults add independent capacity and
a final full-context binding block without modifying that released checkpoint.

- GitHub: [alice-noa-chan/haru](https://github.com/alice-noa-chan/haru)
- Model: [gaon12/haru](https://huggingface.co/gaon12/haru)
- Demo: [gaon12/haru](https://huggingface.co/spaces/gaon12/haru)

Haru is a research prototype for story continuation. It is not an instruction
model, a factual assistant, or a safety-reviewed product.

## Released model at a glance

| Setting | Value |
|---|---:|
| Vocabulary | 8,192 |
| Context length | 512 tokens |
| Local chunk | 64 tokens |
| Hidden size | 384 |
| Query / KV heads | 6 / 2 |
| Physical decoder cells | 2 |
| Recurrent depth | 6 |
| FFN size | 1,024 |
| Trainable parameters | 6,793,363 |
| Training tokens | 800,063,488 |

The two physical cells run in this order:

```text
A -> B -> A -> B -> A -> B
```

The shared language-model head supervises depths 2, 4, and 6. Depth 4 is a
useful CPU setting; depth 6 gives the best measured validation quality.

## Next-training defaults

New training runs use a separate `haru-v2-binding` directory and a separate
12,000-token tokenizer. They do not overwrite the released model's artifacts.

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

model_id = "gaon12/haru"
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

The next-training configuration then applies a non-recurrent full-context
causal attention and SwiGLU block before the shared language-model head. Older
configuration files do not enable this block, so existing exports remain
loadable with their original architecture.

See [RESEARCH.md](RESEARCH.md) for related work.

## Evaluation

The final held-out evaluation used 200 fixed batches at each supervised depth.

| Inference recurrences | Validation loss | Perplexity |
|---:|---:|---:|
| 2 | 2.37096 | 10.708 |
| 4 | 2.06052 | 7.850 |
| 6 | 2.00630 | 7.436 |

Per-token perplexity is comparable only for models using the same tokenizer.
For different tokenizers, use byte- or character-normalized negative
log-likelihood and report actual latency or FLOPs.

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

The the cloud app and persistent Volume are named `haru` and `haru-training`.
a GPU uses fp16 with gradient scaling; a GPU uses bf16.

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
| `generate.py` | Transformers-based local generation |
| `cloud_train.py` | the cloud benchmark, preparation, training, and export |
| `smoke_test.py` | Fast correctness tests |
| `NOTICE.md` | Required training-data attribution |
| `RESEARCH.md` | Related architecture research |

## Known limitations

- Fixed chunk boundaries do not follow sentences or story boundaries.
- Summary memory can lose names, quotations, and exact event details.
- Generation has no KV or recurrent-state cache yet.
- Longer continuations may repeat ideas or drift between entities.
- The repository does not include a parameter-matched Transformer baseline.

## Training data notice

The training corpus is not included in this repository or model release. The
minimal attribution required by its license is recorded in [NOTICE.md](NOTICE.md).

## License

Haru source code and model weights are released under the
[MIT License](LICENSE). The training dataset remains subject to its separate
CC BY 4.0 license described in [NOTICE.md](NOTICE.md).
