---
library_name: transformers
pipeline_tag: text-generation
language: ko
license: mit
tags: [haru, haru-dense, custom-code, story-generation]
---
# Haru v3 Teacher IT

<img src="https://raw.githubusercontent.com/alice-noa-chan/haru/main/assets/haru.png" alt="Haru" width="420" />

The **30,997,377-parameter** instruction-tuned teacher for experimental Korean
children's-story instructions and student distillation. This is the **IT**
version; use `apply_chat_template` rather than treating it as a plain base model.
It is larger than the students' 18M parameter limit. Unquantized FP32 Safetensors
and the included code are **MIT** licensed; see [LICENSE](https://github.com/alice-noa-chan/haru/blob/main/LICENSE).

The teacher base was trained from scratch and frozen before base distillation.
Teacher IT training completed at **10,092,544 tokens**, and the resulting frozen
IT teacher was used to distill the selected Gated8 student IT. Training corpora
are not distributed here. Existing synthetic corpora were reused; no new
external LLM data, teacher outputs or pretrained weights were used.

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
## Chat-formatted story generation

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "alice-noa-chan/haru_3-teacher-chat"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    repo, trust_remote_code=True
).to(device).eval()
messages = [{"role": "user", "content": "토끼가 친구를 만나는 동화를 이어 써 주세요."}]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt", return_dict=True,
).to(device)
with torch.inference_mode():
    output = model.generate(
        **inputs, max_new_tokens=120, do_sample=True,
        temperature=0.8, top_p=0.9, use_cache=True,
    )
print(tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

The chat template includes the training BOS token. `skip_special_tokens=True`
removes assistant role boundaries. Shorten long messages before generation so
the complete input plus output stays within 1,024 tokens.

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
## Evaluation and limitations

Validation story BPC is **0.945599**. On the same 64 held-out raw stories at
context 512, plain continuation BPC is **0.949023**, compared with **0.871838**
for teacher base. Raw-story BPC does not measure instruction following.

On the small program-generated exact-match suite, teacher IT scored **53/70**
with familiar entities/templates, **9/70** with new entities and **0/70** when
both entities and templates changed. This indicates weak generalization.
Actual chat-formatted fixed stories still lose objects, promises and event
continuity. The model is an experimental story checkpoint; reliable factual
assistance and general reasoning have not been established.

See [evaluation.json](https://huggingface.co/alice-noa-chan/haru_3-teacher-chat/blob/main/evaluation.json) for measurements and fixed continuations.
No claim of being the strongest model below 18M is made.

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

## Source and license

The training source is [11f7c7b](https://github.com/alice-noa-chan/haru/tree/11f7c7bddfad1fb92234bc175ffb70499f39616e).
The tokenizer compatibility implementation is recorded at
[8d4a4d1](https://github.com/alice-noa-chan/haru/tree/8d4a4d18b4f87217a7c8ead228fd238e1952b9b2).
Weights and included source code are **MIT** licensed. Raw training corpora are
not distributed with this release.
