---
library_name: transformers
pipeline_tag: text-generation
language: ko
license: mit
tags: [haru, haru-dense, custom-code, story-generation]
---
# Haru v3 Teacher Base (non-IT)

<img src="https://raw.githubusercontent.com/alice-noa-chan/haru/main/assets/haru.png" alt="Haru" width="420" />

A **30,997,377-parameter** self-trained teacher for Korean children's-story
continuation and distillation into compact experimental students. This teacher
is larger than the students' 18M parameter limit. It is a **non-IT / base**
checkpoint: provide the beginning of a story to continue it.

Teacher IT and both evaluated student candidates are now released separately;
see the candidate comparison and repository links below.

The weights are unquantized **FP32 Safetensors**. Weights and included source
code are released under **MIT**; see [LICENSE](https://github.com/alice-noa-chan/haru/blob/main/LICENSE). Training corpora are
not distributed with this repository. Existing synthetic corpora were reused;
no new external-LLM-generated data or pretrained weights were used.

## Installation and compatibility

Use Python 3.11 or newer. CPU inference is supported.

```bash
python -m pip install "torch>=2.6" "transformers==4.57.1" sentencepiece safetensors
```

Independent loading and generation were verified with Transformers **4.57.1
and 5.17.0**. The included wrapper fixes the Transformers 5
`all_tied_weights_keys` loading error. The tokenizer preserves training BOS and
reserved-role tokens under both versions. These compatibility fixes did not
change the trained FP32 weights. `trust_remote_code=True` loads the included
custom model and tokenizer implementation.

Input plus generated output must fit **1,024 tokens**. For GPU inference, use
an appropriate PyTorch build and move both the model and inputs to `cuda`.
## Story continuation

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "alice-noa-chan/haru_3-teacher-base"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    repo, trust_remote_code=True
).to(device).eval()

prompt = "작은 마을에 사는 토끼는 길에서 반짝이는 단추를 발견했어요."
inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=904).to(device)
with torch.inference_mode():
    output = model.generate(
        **inputs, max_new_tokens=120, do_sample=True,
        temperature=0.8, top_p=0.9, repetition_penalty=1.08, use_cache=True,
    )
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

The output contains the input and continuation. To print only new text:

```python
continuation = output[0, inputs["input_ids"].shape[1]:]
print(tokenizer.decode(continuation, skip_special_tokens=True))
```

The example limits the input to 904 tokens and generates at most 120 tokens.
For greedy reproduction, set `do_sample=False` and omit `temperature` and `top_p`.

## Architecture

| Setting | Value |
|---|---|
| Model type | `haru_dense` |
| Decoder layers | 16 |
| Hidden width / FFN width | 384 / 960 |
| Query / KV heads | 6 / 2 (GQA) |
| Vocabulary | 12,000 BPE tokens |
| Context | 1,024 tokens |
| Attention | Full causal attention in every layer; incremental KV cache |
| Position representation | RoPE |
| Normalization | RMSNorm and QK RMSNorm |
| FFN | SwiGLU, independent across teacher layers |
| Attention output | Input-dependent elementwise sigmoid gate |
| Embeddings | Tied input/output embeddings with Korean surface features |

The full-attention v3 decoder differs from the legacy v1/v2 CFRD architecture.
The teacher does not use recurrent cells or shared FFNs.
## Training and checkpoint selection

Base training completed at **2,000,027,648 tokens**. The public weights are the
best story-validation checkpoint at **1,914,437,632 tokens / step 14,606**,
with validation story BPC **0.865567**. The final resume state, including
optimizer, sampler and RNG, is preserved locally.

## Measured performance

The same 64 held-out raw test stories were evaluated at context 512.
Lower BPC is better.

| Model | Parameters | Test story BPC | Relation ranking |
|---|---:|---:|---:|
| Haru v3 Teacher Base | 30,997,377 | 0.871838 | 26/50 |
| Legacy Haru v2 | 16,983,213 | 1.146750 | 23/50 |

Story BPC is approximately 24% lower, but parameter counts, tokenizers and
training mixes differ. This comparison does not isolate the architecture's
effect or establish an advantage for a sub-18M student. Relation ranking remains
close to chance. Three fixed greedy continuations had no repeated 4-grams;
this small observation does not establish broad repetition resistance.
See the [teacher evaluation](https://github.com/alice-noa-chan/haru/blob/main/results/teacher_base_evaluation.json).

## Released models and student candidates

Two students were compared at exactly **13,688,705 parameters** and
**500,039,680 completed distillation tokens** each. Both are publicly available.

| Model | Role | Repository |
|---|---|---|
| Gated8-13m | Selected student, non-IT | [haru_3-student-base](https://huggingface.co/alice-noa-chan/haru_3-student-base) |
| Gated8-13m IT | IT version of the selected student | [haru_3-student-chat](https://huggingface.co/alice-noa-chan/haru_3-student-chat) |
| PairShare8-r48 | Unselected experimental student, non-IT | [haru_3-student-pairshare-base](https://huggingface.co/alice-noa-chan/haru_3-student-pairshare-base) |
| Teacher base | Frozen non-IT teacher | [haru_3-teacher-base](https://huggingface.co/alice-noa-chan/haru_3-teacher-base) |
| Teacher IT | Frozen IT teacher | [haru_3-teacher-chat](https://huggingface.co/alice-noa-chan/haru_3-teacher-chat) |

PairShare shares each SwiGLU FFN across two adjacent layers and adds a rank-48
linear residual adapter per layer. Its attention layers and KV caches remain
independent. Gated8 uses independent FFNs with width 512; PairShare uses width 960.

| Candidate | Validation story BPC (lower is better) | Training-host CPU generation, 4 threads |
|---|---:|---:|
| PairShare8-r48 | 0.945834 | 93.28 tokens/s |
| Gated8-13m | 0.949001 | 110.63 tokens/s |

The paired document-bootstrap 95% interval for PairShare minus Gated validation
BPC was **[-0.009002, 0.003038]**, including zero. The predeclared selection rule
therefore chose the faster CPU model, **Gated8-13m**. Test scores did not drive
selection. This is one training seed; the bootstrap does not measure variation
between training seeds. PairShare is preserved for architecture experiments;
**no PairShare IT checkpoint was trained**.

See the [Haru collection](https://huggingface.co/collections/alice-noa-chan/haru-6abb683d0a8b9677162d7704),
[training instructions](https://github.com/alice-noa-chan/haru/blob/main/TRAINING.md),
and [completed evaluation](https://github.com/alice-noa-chan/haru/blob/main/results/student_final_evaluation.json).

## Limitations

- Generations can lose object locations, promises and event causes.
- Longer outputs may repeat ideas or change characters unexpectedly.
- This base checkpoint has no instruction tuning; use the separate IT teacher for chat-formatted inputs.
- Factual assistance and general reasoning have not been established.
- Contexts longer than 1,024 tokens are unsupported.
- No claim of being the strongest model below 18M is made.

## Source and license

[GitHub](https://github.com/alice-noa-chan/haru) provides the training, distillation
and evaluation code. Weights and included code are **MIT**; training corpora
remain outside this release. The tokenizer compatibility implementation is
recorded at [8d4a4d1](https://github.com/alice-noa-chan/haru/tree/8d4a4d18b4f87217a7c8ead228fd238e1952b9b2).
