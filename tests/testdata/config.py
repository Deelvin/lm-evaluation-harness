import os

API_KEY = os.environ["OCTOAI_API_KEY"]

MODEL_ENDPOINTS = [
    (
        "model_name='llama2-7b-chat-mlc-q0f16'",
        "https://llama2-7b-chat-fp16-1gpu-g2ave3d5t9mm.octoai.run",
    ),
    (
        "model_name='llama2-7b-chat-mlc-q4f16_1'",
        "https://llama2-7b-chat-int4-1gpu-g2ave3d5t9mm.octoai.run",
    ),
    (
        "model_name='llama2-7b-chat-mlc-q8f16_1'",
        "https://llama2-7b-chat-int8-1gpu-g2ave3d5t9mm.octoai.run",
    ),
    (
        "model_name='llama2-13b-chat-mlc-q0f16'",
        "https://llama2-13b-chat-fp16-2gpu-g2ave3d5t9mm.octoai.run",
    ),
    (
        "model_name='llama2-13b-chat-mlc-q4f16_1'",
        "https://llama2-13b-chat-int4-1gpu-g2ave3d5t9mm.octoai.run",
    ),
    (
        "model_name='llama2-13b-chat-mlc-q8f16_1'",
        "https://llama2-13b-chat-int8-1gpu-g2ave3d5t9mm.octoai.run",
    ),
    (
        "model_name='llama2-70b-chat-mlc-q0f16'",
        "https://llama2-70b-chat-fp16-4gpu-g2ave3d5t9mm.octoai.run",
    ),
    (
        "model_name='llama2-70b-chat-mlc-q4f16_1'",
        "https://llama2-70b-chat-int4-2gpu-g2ave3d5t9mm.octoai.run",
    ),
    (
        "model_name='llama2-70b-chat-mlc-q8f16_1'",
        "https://llama2-70b-chat-int8-4gpu-g2ave3d5t9mm.octoai.run",
    ),
]
HEADERS = [
    {
        "accept": "text/event-stream",
        "authorization": f"Bearer {API_KEY}",
        "content-type": "application/json",
    }
]

TASKS = [
    "gsm8k_truncated_7b",
    "triviaqa_truncated_7b",
    "gsm8k_truncated_13b",
    "triviaqa_truncated_13b",
    "gsm8k_truncated_70b",
    "triviaqa_truncated_70b",
]
