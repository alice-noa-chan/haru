"""Export a training checkpoint as a standard Transformers model directory."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

import config
from configuration_cfrd import CFRDConfig
from data_utils import blake2b_file
from model import ModelConfig
from modeling_cfrd import CFRDForCausalLM
from surface_features import build_surface_feature_table
from tokenization_cfrd import CFRDTokenizer
from tokenizer_utils import StoryTokenizer


def select_checkpoint() -> Path:
    """Prefer the best validation checkpoint and fall back to the latest one."""

    path = config.BEST_CHECKPOINT_PATH
    if not path.exists():
        path = config.LATEST_CHECKPOINT_PATH
    if not path.exists():
        raise FileNotFoundError("No trained checkpoint was found")
    return path


def build_model_card(checkpoint: dict, model_config: ModelConfig, parameter_count: int) -> str:
    """Create a small self-contained README for the exported model directory."""

    return f"""---
library_name: transformers
pipeline_tag: text-generation
language: ko
license: mit
tags:
- haru
- cfrd
- custom-code
---

# Haru

Haru is a compact Korean story continuation model built with the custom CFRD
causal architecture. It has {parameter_count:,} parameters and supports
recurrent inference depths 2, 4, and 6.

- [Source code](https://github.com/alice-noa-chan/haru)
- [Interactive demo](https://huggingface.co/spaces/gaon12/haru)

## Usage

Review the included Python files before enabling remote custom code.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "gaon12/haru"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True)

inputs = tokenizer("작은 마을에 아침이 찾아왔어요.", return_tensors="pt")
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

## Model details

- Parameters: {parameter_count:,}
- Context length: {model_config.context_length}
- Recurrent depths supervised during training: {model_config.exit_depths}
- Exported checkpoint step: {int(checkpoint.get('step', 0))}
- Training tokens seen: {int(checkpoint.get('tokens_seen', 0)):,}

## Evaluation

| Recurrent depth | Validation loss | Perplexity |
|---:|---:|---:|
| 2 | 2.37096 | 10.708 |
| 4 | 2.06052 | 7.850 |
| 6 | 2.00630 | 7.436 |

## Training data attribution

[Tiny-Ko-Stories](https://huggingface.co/datasets/psymon/Tiny-Ko-Stories)
by psymon, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The dataset is not redistributed with this model.

## Limitations

- Haru is a continuation model, not an instruction-following assistant.
- Longer generations can repeat ideas or drift between entities.
- The model is not suitable for factual or safety-critical use.
- There is no inference cache yet, so generation recomputes the active context.

## License

Haru model weights and included code are released under the MIT License. The
training dataset remains under its separate CC BY 4.0 license.
"""


@torch.inference_mode()
def verify_export(export_directory, source_model: CFRDForCausalLM, source_tokenizer: CFRDTokenizer) -> None:
    """Reload only through AutoClass APIs and compare tokenization and logits."""

    loaded_tokenizer = AutoTokenizer.from_pretrained(
        export_directory,
        trust_remote_code=True,
        local_files_only=True,
    )
    loaded_model = AutoModelForCausalLM.from_pretrained(
        export_directory,
        trust_remote_code=True,
        local_files_only=True,
    ).eval()

    source_state = source_model.state_dict()
    loaded_state = loaded_model.state_dict()
    if source_state.keys() != loaded_state.keys():
        raise RuntimeError("Exported model state keys differ from the source model")
    for name, source_tensor in source_state.items():
        if not torch.equal(source_tensor, loaded_state[name]):
            raise RuntimeError(f"Exported tensor differs from source model: {name}")

    prompt = config.GENERATION_PROMPT
    source_inputs = source_tokenizer(prompt, return_tensors="pt")
    loaded_inputs = loaded_tokenizer(prompt, return_tensors="pt")
    if not torch.equal(source_inputs["input_ids"], loaded_inputs["input_ids"]):
        raise RuntimeError("Tokenizer IDs changed after export")

    source_logits = source_model(**source_inputs).logits
    loaded_logits = loaded_model(**loaded_inputs).logits
    # Reloading can select a slightly different CPU kernel even when every
    # serialized tensor is bitwise identical. Allow only tiny numerical drift.
    if not torch.allclose(source_logits, loaded_logits, atol=1.0e-5, rtol=1.0e-5):
        max_difference = float((source_logits - loaded_logits).abs().max())
        raise RuntimeError(f"Exported logits differ from source model: {max_difference}")


def main() -> None:
    checkpoint_path = select_checkpoint()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    training_tokenizer = StoryTokenizer()
    training_tokenizer.validate_checkpoint(checkpoint)
    model_config = ModelConfig.from_checkpoint(checkpoint, training_tokenizer.vocab_size)
    hf_config = CFRDConfig.from_model_config(
        model_config,
        inference_recurrences=config.INFERENCE_RECURSIONS,
        bos_token_id=training_tokenizer.bos_id,
        eos_token_id=training_tokenizer.eos_id,
        pad_token_id=training_tokenizer.pad_id,
    )

    hf_model = CFRDForCausalLM(hf_config)
    state_dict = dict(checkpoint["model"])
    if "surface_feature_table" not in state_dict:
        state_dict["surface_feature_table"] = build_surface_feature_table(training_tokenizer)
    hf_model.model.load_state_dict(state_dict)
    hf_model.eval()

    hf_tokenizer = CFRDTokenizer(
        vocab_file=str(config.TOKENIZER_MODEL_PATH),
        newline_token=config.NEWLINE_TOKEN,
        newline_escape_token=config.NEWLINE_ESCAPE_TOKEN,
        bos_token=training_tokenizer.id_to_piece(training_tokenizer.bos_id),
        eos_token=training_tokenizer.id_to_piece(training_tokenizer.eos_id),
        unk_token=training_tokenizer.id_to_piece(training_tokenizer.unk_id),
        pad_token=training_tokenizer.id_to_piece(training_tokenizer.pad_id),
        model_max_length=model_config.context_length,
        padding_side="left",
    )

    export_directory = config.TRANSFORMERS_EXPORT_DIR
    export_directory.mkdir(parents=True, exist_ok=True)
    hf_model.save_pretrained(export_directory, safe_serialization=True)
    hf_tokenizer.save_pretrained(export_directory)
    shutil.copy2(config.CODE_DIR / "LICENSE", export_directory / "LICENSE")

    generation_config = GenerationConfig(
        max_new_tokens=config.GENERATION_MAX_NEW_TOKENS,
        do_sample=config.GENERATION_TEMPERATURE > 0.0,
        temperature=max(config.GENERATION_TEMPERATURE, 1.0e-5),
        top_p=config.GENERATION_TOP_P,
        top_k=config.GENERATION_TOP_K,
        repetition_penalty=config.GENERATION_REPETITION_PENALTY,
        bos_token_id=training_tokenizer.bos_id,
        eos_token_id=training_tokenizer.eos_id,
        pad_token_id=training_tokenizer.pad_id,
        use_cache=False,
    )
    generation_config.save_pretrained(export_directory)

    (export_directory / "README.md").write_text(
        build_model_card(
            checkpoint,
            model_config,
            sum(parameter.numel() for parameter in hf_model.parameters()),
        ),
        encoding="utf-8",
    )
    (export_directory / "export_metadata.json").write_text(
        json.dumps(
            {
                "source_checkpoint": str(checkpoint_path.relative_to(config.ROOT_DIR)),
                "source_checkpoint_step": int(checkpoint.get("step", 0)),
                "tokenizer_blake2b16": blake2b_file(config.TOKENIZER_MODEL_PATH),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    verify_export(export_directory, hf_model, hf_tokenizer)
    print(f"Transformers export verified: {export_directory}", flush=True)


if __name__ == "__main__":
    main()
