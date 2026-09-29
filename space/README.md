---
title: Haru
emoji: 🌸
colorFrom: pink
colorTo: indigo
sdk: gradio
sdk_version: 6.22.0
app_file: app.py
pinned: false
license: mit
models:
- alice-noa-chan/haru_2
---

# Haru

CPU demo for Haru Korean continuation and chat models. The verified v3
student-chat checkpoint is selected when released; until then the preserved
[Haru v2](https://huggingface.co/alice-noa-chan/haru_2) remains available.

The demo displays the selected model and actual parameter count. Recurrent
depth controls have been removed. Input plus generation must fit the model
context; the application asks for a shorter input instead of silently truncating.
