import os

API_KEY = os.environ["OCTOAI_TOKEN"]

ENDPOINTS_DATA = {
      "llama-2-13b-chat-fp16": 2,
      "llama-2-70b-chat-int4": 3,
      "llama-2-70b-chat-fp16": 3,
      "codellama-7b-instruct-fp16": 2,
      "codellama-13b-instruct-fp16": 3,
      "codellama-34b-instruct-int4": 4,
      "codellama-34b-instruct-fp16": 5,
      "mistral-7b-instruct-fp16": 3
    }

HEADERS = [
    {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {API_KEY}'
    }
]

TASKS = [
    "gsm8k_truncated_llama",
    "triviaqa_truncated_llama",
    "gsm8k_truncated_codellama",
    "triviaqa_truncated_codellama",
    "gsm8k_truncated_mistral",
    "triviaqa_truncated_mistral"
]