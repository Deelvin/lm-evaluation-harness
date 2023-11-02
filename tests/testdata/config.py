import os

API_KEY=os.environ["OCTOAI_API_KEY"]

MODEL_ENDPOINTS = [
    # "https://llama2-7b-chat-fp16-1gpu-g2ave3d5t9mm.octoai.run",
    # "https://codellama-7b-instruct-fp16-1gpu-g2ave3d5t9mm.octoai.run",
    # "https://codellama-7b-instruct-int4-1gpu-g2ave3d5t9mm.octoai.run",
    # "https://codellama-7b-instruct-int8-1gpu-g2ave3d5t9mm.octoai.run",
    # "https://codellama-13b-instruct-fp16-2gpu-g2ave3d5t9mm.octoai.run",
    # "https://codellama-13b-instruct-int4-1gpu-g2ave3d5t9mm.octoai.run",
    # "https://codellama-13b-instruct-int8-1gpu-g2ave3d5t9mm.octoai.run",
    # "https://codellama-34b-instruct-fp16-4gpu-g2ave3d5t9mm.octoai.run",
    # "https://codellama-34b-instruct-int4-1gpu-g2ave3d5t9mm.octoai.run",
    # "https://codellama-34b-instruct-int8-2gpu-g2ave3d5t9mm.octoai.run",
    ("model_name='llama2-7b-chat-mlc-q0f16'", "https://llama2-7b-chat-fp16-1gpu-g2ave3d5t9mm.octoai.run"),
    ("model_name='llama2-7b-chat-mlc-q4f16_1'", "https://llama2-7b-chat-int4-1gpu-g2ave3d5t9mm.octoai.run"),
    ("model_name='llama2-7b-chat-mlc-q8f16_1'", "https://llama2-7b-chat-int8-1gpu-g2ave3d5t9mm.octoai.run"),
    ("model_name='llama2-13b-chat-mlc-q0f16'", "https://llama2-13b-chat-fp16-2gpu-g2ave3d5t9mm.octoai.run"),
    ("model_name='llama2-13b-chat-mlc-q4f16_1'", "https://llama2-13b-chat-int4-1gpu-g2ave3d5t9mm.octoai.run"),
    ("model_name='llama2-13b-chat-mlc-q8f16_1'", "https://llama2-13b-chat-int8-1gpu-g2ave3d5t9mm.octoai.run"),
    # "https://llama2-70b-chat-fp16-4gpu-g2ave3d5t9mm.octoai.run",
    # "https://llama2-70b-chat-int4-2gpu-g2ave3d5t9mm.octoai.run",
    # "https://llama2-70b-chat-int8-4gpu-g2ave3d5t9mm.octoai.run",
    # TODO(vvchernov): it is demo, may be need to remove
    #"https://llama-2-70b-chat-demo-kk0powt97tmb.octoai.run",
]
HEADERS = [{
    "accept": "text/event-stream",
    "authorization": f"Bearer {API_KEY}",
    "content-type": "application/json",
}]

TASKS = [
    "gsm8k",
    #"gsm8k_truncated"
    #"triviaqa",
    # "qasper",
    # "squad2",
    # "truthfulqa",
    #"mgsm"
]