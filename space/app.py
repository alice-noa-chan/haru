"""Gradio demo for the Haru Korean story continuation model."""

from __future__ import annotations

import os

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.environ.get("HARU_MODEL_ID", "gaon12/haru")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = (
    AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
    )
    .cpu()
    .eval()
)


@torch.inference_mode()
def continue_story(
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    recurrences: int,
    seed: int,
) -> str:
    """Continue one Korean opening with user-selected sampling settings."""

    prompt = prompt.strip()
    if not prompt:
        raise gr.Error("Please enter a Korean story opening.")

    torch.manual_seed(int(seed))
    model.config.inference_recurrences = int(recurrences)
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(
        **inputs,
        max_new_tokens=int(max_new_tokens),
        do_sample=True,
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=40,
        repetition_penalty=1.08,
        use_cache=False,
        bad_words_ids=[
            [tokenizer.pad_token_id],
            [tokenizer.bos_token_id],
            [tokenizer.unk_token_id],
        ],
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)


DESCRIPTION = """
# Haru 🌸

A compact Korean story continuation model with 6.8 million parameters.
Write an opening sentence, adjust the controls, and let Haru continue it.

Haru is a research prototype. Review generated text before sharing it.
"""

CSS = """
.gradio-container { max-width: 980px !important; }
#story-output textarea { font-size: 1.05rem; line-height: 1.75; }
"""

with gr.Blocks(title="Haru") as demo:
    gr.Markdown(DESCRIPTION)
    prompt = gr.Textbox(
        label="Story opening",
        value="작은 마을에 조용한 아침이 찾아왔어요.",
        lines=3,
        placeholder="한국어 이야기의 첫 문장을 입력하세요.",
    )

    with gr.Row():
        max_new_tokens = gr.Slider(
            minimum=32,
            maximum=200,
            value=120,
            step=8,
            label="Maximum new tokens",
        )
        recurrences = gr.Radio(
            choices=[2, 4, 6],
            value=6,
            label="Recurrent depth",
            info="4 is faster; 6 gives the best measured quality.",
        )

    with gr.Row():
        temperature = gr.Slider(0.2, 1.2, value=0.7, step=0.05, label="Temperature")
        top_p = gr.Slider(0.5, 1.0, value=0.9, step=0.01, label="Top-p")
        seed = gr.Number(value=42, precision=0, label="Seed")

    generate_button = gr.Button("Continue story", variant="primary")
    output = gr.Textbox(
        label="Haru's continuation",
        lines=14,
        buttons=["copy"],
        elem_id="story-output",
    )

    gr.Examples(
        examples=[
            ["작은 마을에 조용한 아침이 찾아왔어요."],
            ["하린이는 비 오는 날 작은 강아지를 발견했어요."],
            ["깊은 산속 우체통에 반짝이는 편지 한 통이 도착했어요."],
        ],
        inputs=[prompt],
    )

    generate_button.click(
        fn=continue_story,
        inputs=[prompt, max_new_tokens, temperature, top_p, recurrences, seed],
        outputs=output,
        api_name="continue_story",
        concurrency_limit=1,
    )
    prompt.submit(
        fn=continue_story,
        inputs=[prompt, max_new_tokens, temperature, top_p, recurrences, seed],
        outputs=output,
        concurrency_limit=1,
    )

if __name__ == "__main__":
    demo.queue(max_size=8).launch(css=CSS)
