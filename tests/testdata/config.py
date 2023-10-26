import os

API_KEY=os.environ["OCTOAI_API_KEY"]

MODEL_NAMES = [
    #"codellama-7b-instruct-mlc-q0f16",
    "codellama-7b-instruct-mlc-q4f16_1",
    "codellama-7b-instruct-mlc-q8f16_1",
    "codellama-13b-instruct-mlc-q0f16",
    "codellama-13b-instruct-mlc-q4f16_1",
    "codellama-13b-instruct-mlc-q8f16_1",
    "codellama-34b-instruct-mlc-q0f16",
    "codellama-34b-instruct-mlc-q4f16_1",
    "codellama-34b-instruct-mlc-q8f16_1",
    "llama2-7b-chat-mlc-q0f16",
    "llama2-7b-chat-mlc-q4f16_1",
    "llama2-7b-chat-mlc-q8f16_1",
    "llama2-13b-chat-mlc-q0f16",
    "llama2-13b-chat-mlc-q4f16_1",
    "llama2-13b-chat-mlc-q8f16_1",
    "llama2-70b-chat-mlc-q0f16",
    "llama2-70b-chat-mlc-q4f16_1",
    "llama2-70b-chat-mlc-q8f16_1",
    # TODO(vvchernov): it is demo, may be need to remove
    "llama-2-70b-chat",
]

MODEL_ENDPOINTS = [
    "https://llama2-7b-chat-fp16-1gpu-g2ave3d5t9mm.octoai.run",
    "https://codellama-7b-instruct-fp16-1gpu-g2ave3d5t9mm.octoai.run",
    "https://codellama-7b-instruct-int4-1gpu-g2ave3d5t9mm.octoai.run",
    "https://codellama-7b-instruct-int8-1gpu-g2ave3d5t9mm.octoai.run",
    "https://codellama-13b-instruct-fp16-2gpu-g2ave3d5t9mm.octoai.run",
    "https://codellama-13b-instruct-int4-1gpu-g2ave3d5t9mm.octoai.run",
    "https://codellama-13b-instruct-int8-1gpu-g2ave3d5t9mm.octoai.run",
    "https://codellama-34b-instruct-fp16-4gpu-g2ave3d5t9mm.octoai.run",
    "https://codellama-34b-instruct-int4-1gpu-g2ave3d5t9mm.octoai.run",
    "https://codellama-34b-instruct-int8-2gpu-g2ave3d5t9mm.octoai.run",
    "https://llama2-7b-chat-fp16-1gpu-g2ave3d5t9mm.octoai.run",
    "https://llama2-7b-chat-int4-1gpu-g2ave3d5t9mm.octoai.run",
    "https://llama2-7b-chat-int8-1gpu-g2ave3d5t9mm.octoai.run",
    "https://llama2-13b-chat-fp16-2gpu-g2ave3d5t9mm.octoai.run",
    "https://llama2-13b-chat-int4-1gpu-g2ave3d5t9mm.octoai.run",
    "https://llama2-13b-chat-int8-1gpu-g2ave3d5t9mm.octoai.run",
    "https://llama2-70b-chat-fp16-4gpu-g2ave3d5t9mm.octoai.run",
    "https://llama2-70b-chat-int4-2gpu-g2ave3d5t9mm.octoai.run",
    "https://llama2-70b-chat-int8-4gpu-g2ave3d5t9mm.octoai.run",
    # TODO(vvchernov): it is demo, may be need to remove
    "https://llama-2-70b-chat-demo-kk0powt97tmb.octoai.run",
]
HEADERS = [{
    "accept": "text/event-stream",
    "authorization": f"Bearer {API_KEY}",
    "content-type": "application/json",
}]

TASKS = [
    "gsm8k",
    "triviaqa",
    "qasper",
    "squad2",
    "truthfulqa",
    #"mgsm"
]