"""CPU demo: v3 chat when released, preserved v2 continuation meanwhile."""

from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
import torch
from huggingface_hub import HfApi
from huggingface_hub.errors import RepositoryNotFoundError
from transformers import AutoModelForCausalLM, AutoTokenizer


def default_model():
    candidate = "alice-noa-chan/haru_3-student-chat"
    try:
        info = HfApi().model_info(candidate)
        if any(file.rfilename == "model.safetensors" for file in info.siblings):
            return candidate
    except RepositoryNotFoundError:
        pass
    return "alice-noa-chan/haru_2"


MODEL_ID = os.environ.get("HARU_MODEL_ID") or default_model()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True).cpu().eval()
IS_CHAT = model.config.model_type == "haru_dense" and MODEL_ID.endswith("-chat")
CONTEXT = getattr(model.config, "max_position_embeddings", getattr(model.config, "context_length", 512))
PARAMETERS = sum(parameter.numel() for parameter in model.parameters())


@torch.inference_mode()
def respond(prompt, max_new_tokens, temperature, top_p, seed):
    prompt = prompt.strip()
    if not prompt:
        raise gr.Error("한국어 질문이나 이야기의 첫 문장을 입력해 주세요.")
    torch.manual_seed(int(seed))
    if IS_CHAT:
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
    else:
        inputs = tokenizer(prompt, return_tensors="pt")
    if inputs["input_ids"].shape[1] + int(max_new_tokens) > CONTEXT:
        raise gr.Error(f"입력과 생성 길이의 합이 {CONTEXT} 토큰을 넘습니다. 입력 또는 생성 길이를 줄여 주세요.")
    output = model.generate(
        **inputs,
        max_new_tokens=int(max_new_tokens),
        do_sample=True,
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=40,
        repetition_penalty=1.08,
        use_cache=model.config.model_type == "haru_dense",
        bad_words_ids=[[tokenizer.pad_token_id], [tokenizer.bos_token_id], [tokenizer.unk_token_id]],
    )
    completion = output[0, inputs["input_ids"].shape[1] :] if IS_CHAT else output[0]
    return tokenizer.decode(completion, skip_special_tokens=True)


with gr.Blocks(title="Haru") as demo:
    gr.Image(
        value=str(Path(__file__).parent / "assets/haru.png"),
        show_label=False,
        height=260,
        interactive=False,
        show_download_button=False,
    )
    gr.Markdown(
        f"# Haru 🌸\n한국어 {'대화' if IS_CHAT else '이야기 이어쓰기'} 연구 모델 · {PARAMETERS:,} parameters\n\n사용 중인 모델: [{MODEL_ID}](https://huggingface.co/{MODEL_ID})"
    )
    prompt = gr.Textbox(label="질문 또는 이야기의 시작", value="작은 마을에 조용한 아침이 찾아왔어요.", lines=3)
    with gr.Row():
        length = gr.Slider(16, 200, value=120, step=8, label="생성 토큰 수")
        temperature = gr.Slider(0.2, 1.2, value=0.7, step=0.05, label="Temperature")
        top_p = gr.Slider(0.5, 1.0, value=0.9, step=0.01, label="Top-p")
        seed = gr.Number(value=42, precision=0, label="Seed")
    button = gr.Button("생성", variant="primary")
    output = gr.Textbox(label="Haru의 답변", lines=14)
    controls = [prompt, length, temperature, top_p, seed]
    button.click(respond, inputs=controls, outputs=output, concurrency_limit=1)
    prompt.submit(respond, inputs=controls, outputs=output, concurrency_limit=1)
if __name__ == "__main__":
    demo.queue(max_size=8).launch()
