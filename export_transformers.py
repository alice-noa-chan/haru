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
from model_factory import CFRD_ARCH, architecture_of_checkpoint
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


def load_evaluation_results() -> dict[str, float | int]:
    """Load metrics produced by evaluate.py for the active run, if present."""

    path = config.RUN_DIR / "evaluation.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Evaluation output must be a JSON object: {path}")
    return payload


def build_evaluation_table(model_config: ModelConfig, evaluation: dict[str, float | int]) -> str:
    """Render only metrics that belong to the checkpoint's active run."""

    rows = []
    for depth in model_config.exit_depths:
        loss = evaluation.get(f"depth_{depth}_loss")
        perplexity = evaluation.get(f"depth_{depth}_perplexity")
        if loss is None or perplexity is None:
            continue
        rows.append(f"| {depth} | {float(loss):.5f} | {float(perplexity):.3f} |")
    if not rows:
        return "Evaluation metrics are not included in this export. Run `python evaluate.py` before exporting."
    return "\n".join(
        [
            "| Recurrent depth | Validation loss | Perplexity |",
            "|---:|---:|---:|",
            *rows,
        ]
    )


def build_model_card(
    checkpoint: dict,
    model_config: ModelConfig,
    parameter_count: int,
    evaluation: dict[str, float | int],
) -> str:
    """Create a small self-contained README for the exported model directory."""

    # Both claims used to be fixed strings that described v1.x. Published for
    # v2.0 they were false and mutually contradictory: the header promised
    # depths 2, 4 and 6 while the evaluation table below it showed depth 2 at
    # perplexity 6605. Early exit only works when the intermediate exits are
    # trained, so derive the claim from the weight that trains them.
    supervised = config.AUX_EXIT_LOSS_WEIGHT > 0.0
    depths = model_config.exit_depths
    if supervised and len(depths) > 1:
        listed = ", ".join(str(d) for d in depths[:-1])
        depth_claim = f"recurrent inference depths {listed} and {depths[-1]}"
        supervised_depths = str(depths)
    else:
        depth_claim = f"recurrent depth {model_config.recurrences} only"
        supervised_depths = (
            f"none; AUX_EXIT_LOSS_WEIGHT is {config.AUX_EXIT_LOSS_WEIGHT}, so the "
            f"depth {' and '.join(str(d) for d in depths[:-1])} exits are untrained"
        )

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

# Haru v{config.RELEASE_VERSION}

Haru is a compact Korean story continuation model built with the custom CFRD
causal architecture. It has {parameter_count:,} parameters and runs at
{depth_claim}.

- [Source code](https://github.com/alice-noa-chan/haru)
- [Interactive demo](https://huggingface.co/spaces/gaon12/haru)
- [Previous Haru release](https://huggingface.co/{config.PREVIOUS_HUGGINGFACE_MODEL_ID})

## Usage

Review the included Python files before enabling remote custom code.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "{config.HUGGINGFACE_MODEL_ID}"
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
- Recurrent depths supervised during training: {supervised_depths}
- Exported checkpoint step: {int(checkpoint.get("step", 0))}
- Training tokens seen: {int(checkpoint.get("tokens_seen", 0)):,}

## Evaluation

{build_evaluation_table(model_config, evaluation)}

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

    # ModelConfig.from_checkpoint keeps only the fields it recognizes, and every
    # field except vocab_size has a default. A dense-baseline checkpoint would
    # therefore build a default CFRD and fail later on a state_dict mismatch
    # that says nothing about the real cause.
    architecture = architecture_of_checkpoint(checkpoint)
    if architecture != CFRD_ARCH:
        raise ValueError(
            f"{checkpoint_path} holds a {architecture} model. The Transformers export format is CFRD-only; "
            "ablation baselines are controls and are not released."
        )

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
            load_evaluation_results(),
        ),
        encoding="utf-8",
    )
    (export_directory / "export_metadata.json").write_text(
        json.dumps(
            {
                "source_checkpoint": str(checkpoint_path.relative_to(config.ROOT_DIR)),
                "source_checkpoint_step": int(checkpoint.get("step", 0)),
                "tokenizer_blake2b16": blake2b_file(config.TOKENIZER_MODEL_PATH),
                "release_version": config.RELEASE_VERSION,
                "huggingface_model_id": config.HUGGINGFACE_MODEL_ID,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    verify_export(export_directory, hf_model, hf_tokenizer)
    print(f"Transformers export verified: {export_directory}", flush=True)


if __name__ == "__main__":
    main()
