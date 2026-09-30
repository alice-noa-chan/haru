"""Export a tested v3 checkpoint; never substitute random weights for a release."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import asdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from configuration_haru_dense import HaruDenseConfig
from data_utils import blake2b_file
from dense_model import parameter_count
from haru.data import CHAT_TEMPLATE, ROLES
from haru.runtime import atomic_json
from haru.training import load_model
from modeling_haru_dense import HaruDenseForCausalLM
from tokenization_cfrd import CFRDTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def export(checkpoint_path, tokenizer_path, output, repo_id=None, copy_resume=True, public_source_commit=None):
    model, checkpoint = load_model(checkpoint_path)
    if checkpoint["tokenizer_blake2b16"] != blake2b_file(tokenizer_path):
        raise ValueError("Export tokenizer differs from training tokenizer")
    if public_source_commit is not None and not re.fullmatch(r"[0-9a-f]{40}", public_source_commit):
        raise ValueError("Public source commit must be a full 40-character Git SHA")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    wrapper = HaruDenseForCausalLM(HaruDenseConfig(dense_config=asdict(model.cfg)))
    wrapper.model.load_state_dict(model.state_dict(), strict=True)
    wrapper.float().eval()
    wrapper.generation_config = GenerationConfig(bos_token_id=2, eos_token_id=3, pad_token_id=0, use_cache=True)
    tokenizer = CFRDTokenizer(
        str(tokenizer_path),
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
        model_max_length=model.cfg.context_length,
        additional_special_tokens=ROLES,
        chat_template=CHAT_TEMPLATE,
        padding_side="left",
    )
    if len(tokenizer) != model.cfg.vocab_size:
        raise ValueError("Chat symbols were not reserved inside the 12K vocabulary")
    if checkpoint.get("chat_template"):
        wrapper.generation_config.eos_token_id = [3, tokenizer.convert_tokens_to_ids("<|end|>")]
    wrapper.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    # Explicitly ship all source dependencies; do not depend on a local import cache.
    for name in ("dense_model.py", "configuration_haru_dense.py", "modeling_haru_dense.py", "tokenization_cfrd.py"):
        shutil.copyfile(PROJECT_ROOT / name, output / name)
    shutil.copyfile(PROJECT_ROOT / "LICENSE", output / "LICENSE")
    (output / "assets").mkdir(exist_ok=True)
    shutil.copyfile(PROJECT_ROOT / "assets/haru.png", output / "assets/haru.png")
    loaded_tokenizer = AutoTokenizer.from_pretrained(output, trust_remote_code=True, local_files_only=True)
    loaded = AutoModelForCausalLM.from_pretrained(output, trust_remote_code=True, local_files_only=True).eval()
    prompt = loaded_tokenizer("빨간 열쇠는 왼쪽 서랍에 있습니다.", return_tensors="pt")
    with torch.no_grad():
        torch.testing.assert_close(wrapper(**prompt).logits, loaded(**prompt).logits, atol=1e-5, rtol=1e-5)
    for key, tensor in wrapper.state_dict().items():
        if not torch.equal(tensor, loaded.state_dict()[key]):
            raise RuntimeError(f"Export changed {key}")
    if copy_resume:
        shutil.copyfile(checkpoint_path, output / "training_state.pt")
    metadata = {
        "parameters": parameter_count(model.cfg),
        "model_config": asdict(model.cfg),
        "tokens_seen": checkpoint["tokens_seen"],
        "step": checkpoint["step"],
        "source_commit": public_source_commit or checkpoint["source_commit"],
        "tokenizer_blake2b16": checkpoint["tokenizer_blake2b16"],
        "data_manifest_blake2b16": checkpoint["data_manifest_blake2b16"],
        "phase": checkpoint["phase"],
        "instruction_tuned": bool(checkpoint.get("chat_template")),
        "variant": "it" if checkpoint.get("chat_template") else "non-it",
        "role": "teacher" if checkpoint["phase"].startswith("teacher") else "student",
        "objective": checkpoint.get("objective", "general_korean"),
        "checkpoint_blake2b16": blake2b_file(checkpoint_path),
        "weight_dtype": "float32",
        "teacher_checkpoint_blake2b16": checkpoint.get("training_config", {}).get("teacher_blake2b16"),
        "export_verified": True,
    }
    atomic_json(output / "export_metadata.json", metadata)
    model_id = repo_id or "local-export"
    from haru.training import checkpoint_validation

    evaluation = checkpoint_validation(checkpoint_path, checkpoint)
    if evaluation:
        atomic_json(output / "validation.json", evaluation)
    story_first = metadata["objective"] == "story_continuation"
    role_note = (
        "This larger teacher supports distillation into a sub-18M student; the student parameter limit does not apply to it."
        if metadata["role"] == "teacher"
        else "This is the compact student model."
    )
    purpose = (
        "This model is trained for Korean children's-story continuation. General-language "
        "corpora support fluency, but story validation drives checkpoint selection."
        if story_first
        else "This model is a compact Korean language-model research checkpoint."
    )
    validation_label = "Story BPC" if story_first else "Macro BPC"
    validation_value = (
        evaluation["domains"]["story"]["bpc"]
        if story_first and evaluation
        else (evaluation["macro_bpc"] if evaluation else "not yet recorded")
    )
    architecture_note = (
        f"FFNs are shared across groups of {model.cfg.ffn_share_group_size} adjacent layers; "
        f"each layer has a rank-{model.cfg.ffn_adapter_rank} linear residual adapter. "
        "Attention and KV caches remain independent per logical layer."
        if model.cfg.ffn_share_group_size > 1
        else f"FFNs are independent per layer; linear adapter rank: {model.cfg.ffn_adapter_rank}."
    )
    card = f"""---
library_name: transformers
pipeline_tag: text-generation
language: ko
license: mit
tags: [haru, haru-dense, custom-code]
---
# {model_id}

<img src="assets/haru.png" alt="Haru" width="420" />

Haru v3 {checkpoint["phase"]}: {metadata["parameters"]:,} parameters, context 1024,
full causal attention, QK RMSNorm, SwiGLU and an incremental KV cache.
{architecture_note}
Training tokens: {checkpoint["tokens_seen"]:,}. Full FP32 weights are preserved.
Variant: **{metadata["variant"]}**. {role_note}
{purpose}

Source: [alice-noa-chan/haru](https://github.com/alice-noa-chan/haru/tree/{metadata["source_commit"]}).
Public implementation commit: `{metadata["source_commit"]}`.
Teacher/student training uses the same tokenizer. No new external LLM data,
teacher logits or pretrained model weights were used. Existing synthetic corpora
were reused. Training corpora are not distributed with this model.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
repo = {model_id!r}
tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True).eval()
inputs = tokenizer("작은 마을에 아침이 찾아왔어요.", return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=120, use_cache=True)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

IT (chat) checkpoints also support `tokenizer.apply_chat_template(messages,
add_generation_prompt=True, return_tensors="pt")`. Base checkpoints continue
text and are non-IT. Input plus generation must fit 1024 tokens.

Model weights and included source code are licensed under MIT (see LICENSE).
Training-source records and raw corpora are kept outside the public release.
{("`training_state.pt` preserves optimizer, schedule, sampler and RNG states for trusted-source training resumption." if copy_resume else "")}
Validation {validation_label}: {validation_value}.
Public benchmark comparisons and limitations are documented in the source repository.
No claim of being the strongest under 18M is made without matching evidence.
"""
    (output / "README.md").write_text(card, encoding="utf-8")
    if repo_id:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(repo_id, private=False, exist_ok=True)
        api.upload_folder(
            repo_id=repo_id, folder_path=output, commit_message=f"Release verified Haru v3 {checkpoint['phase']}"
        )
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=Path("packed/haru-v3-story/tokenizer.model"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id")
    parser.add_argument("--public-source-commit", help="Equivalent public commit after a source-layout migration")
    args = parser.parse_args()
    print(
        json.dumps(
            export(
                args.checkpoint,
                args.tokenizer,
                args.output,
                args.repo_id,
                public_source_commit=args.public_source_commit,
            ),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
