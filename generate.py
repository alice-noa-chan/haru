"""Generate text through the exported Transformers AutoClass interface."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import config
from train import resolve_device


@torch.inference_mode()
def main() -> None:
    export_directory = config.TRANSFORMERS_EXPORT_DIR
    if not (export_directory / "config.json").exists():
        raise FileNotFoundError(
            f"No Transformers export found at {export_directory}. "
            "Run `python export_transformers.py` first."
        )

    device = resolve_device()
    torch.manual_seed(config.GENERATION_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.GENERATION_SEED)

    tokenizer = AutoTokenizer.from_pretrained(
        export_directory,
        trust_remote_code=True,
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        export_directory,
        trust_remote_code=True,
        local_files_only=True,
    ).to(device)
    model.eval()
    model.config.inference_recurrences = config.INFERENCE_RECURSIONS

    inputs = tokenizer(config.GENERATION_PROMPT, return_tensors="pt").to(device)
    generation_arguments = {
        "max_new_tokens": config.GENERATION_MAX_NEW_TOKENS,
        "do_sample": config.GENERATION_TEMPERATURE > 0.0,
        "top_p": config.GENERATION_TOP_P,
        "top_k": config.GENERATION_TOP_K,
        "repetition_penalty": config.GENERATION_REPETITION_PENALTY,
        "use_cache": False,
        "bad_words_ids": [
            [tokenizer.pad_token_id],
            [tokenizer.bos_token_id],
            [tokenizer.unk_token_id],
        ],
    }
    if config.GENERATION_TEMPERATURE > 0.0:
        generation_arguments["temperature"] = config.GENERATION_TEMPERATURE

    output_ids = model.generate(**inputs, **generation_arguments)
    print(tokenizer.decode(output_ids[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
