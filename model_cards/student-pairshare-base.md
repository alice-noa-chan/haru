---
library_name: transformers
pipeline_tag: text-generation
language: ko
license: mit
tags: [haru, haru-dense, custom-code, story-generation]
---
# Haru v3 PairShare8-r48 — experimental student non-IT

<img src="https://raw.githubusercontent.com/alice-noa-chan/haru/main/assets/haru.png" alt="Haru" width="420" />

The **13,688,705-parameter PairShare8-r48** candidate for Korean children's-story
continuation. This independent repository preserves the **unselected experimental
student**, including completed weights, usage, evaluation and architecture.
It is a **non-IT / base** model. No PairShare IT model was trained.

These are the same unquantized **FP32** weights previously published on the
[`pairshare8-r48` branch](https://huggingface.co/alice-noa-chan/haru_3-student-base/tree/pairshare8-r48),
not a retrained checkpoint. The selected default is
[Gated8 student non-IT](https://huggingface.co/alice-noa-chan/haru_3-student-base).

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

repo = "alice-noa-chan/haru_3-student-pairshare-base"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    repo, trust_remote_code=True
).to(device).eval()
inputs = tokenizer(
    "작은 마을에 사는 토끼는 길에서 반짝이는 단추를 발견했어요.",
    return_tensors="pt", truncation=True, max_length=904,
).to(device)
with torch.inference_mode():
    output = model.generate(
        **inputs, max_new_tokens=120, do_sample=True,
        temperature=0.8, top_p=0.9, repetition_penalty=1.08, use_cache=True,
    )
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

No branch or special revision argument is needed for this independent repository.
For only the continuation, decode `output[0, inputs["input_ids"].shape[1]:]`.

## Experimental architecture

| Setting | Value |
|---|---|
| Model type | `haru_dense` |
| Parameters | 13,688,705 |
| Logical decoder layers | 8 |
| Hidden width / SwiGLU width | 384 / 960 |
| FFN sharing | One FFN per adjacent pair of layers (4 shared groups) |
| Layer correction | Independent rank-48 linear residual adapter per layer |
| Attention / KV cache | Independent per logical layer |
| Query / KV heads | 6 / 2 (GQA) |
| Vocabulary / context | 12,000 BPE / 1,024 tokens |
| Other components | RMSNorm, QK RMSNorm, RoPE, sigmoid attention output gate |
| Embeddings | Tied input/output embeddings with Korean surface features |

## Training and measured results

Both student candidates completed **500,039,680 distillation tokens**, using
**0.5 CE + 0.5 × T² KL(teacher || student), T=2**, with a frozen self-trained
teacher. The released PairShare weights are the best story-validation checkpoint
at **475,398,144 tokens / step 3,627**; completed training and selected checkpoint
token counts are intentionally distinguished.

Test story BPC was **0.952942** on the same 64 held-out raw stories at context 512;
the selected Gated8 scored **0.960557**. Test scores did not change the
predeclared validation-plus-CPU selection. See [evaluation.json](https://huggingface.co/alice-noa-chan/haru_3-student-pairshare-base/blob/main/evaluation.json)
for the actual fixed continuations and [validation.json](https://huggingface.co/alice-noa-chan/haru_3-student-pairshare-base/blob/main/validation.json) for
validation records.

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

## Limitations and license

- Experimental FFN sharing is not established as better than the independent control.
- Only one training seed was used.
- Fixed generations can replace the original objects with repeated spoons and lose book-return promises.
- Relation generalization remains weak; reliable reasoning or factual assistance has not been established.
- The model is non-IT and cannot use contexts longer than 1,024 tokens.
- No claim of being the strongest model below 18M is made.

Weights and included source code are **MIT** licensed; see [LICENSE](https://github.com/alice-noa-chan/haru/blob/main/LICENSE).
Training corpora are not redistributed. Existing synthetic corpora were reused;
no new external LLM data, teacher outputs or pretrained weights were used.
The distillation teacher was trained within this project.

Training implementation: [11f7c7b](https://github.com/alice-noa-chan/haru/tree/11f7c7bddfad1fb92234bc175ffb70499f39616e).
Tokenizer compatibility implementation: [8d4a4d1](https://github.com/alice-noa-chan/haru/tree/8d4a4d18b4f87217a7c8ead228fd238e1952b9b2).
