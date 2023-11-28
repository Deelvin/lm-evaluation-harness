import os

API_KEY = os.environ["OCTOAI_API_KEY"]

ENDPOINTS_DATA = [
    {
      "llama":
          [
            {
              "url": "https://text.customer-endpoints.nimbus.octoml.ai",
              "model": "llama-2-13b-chat-fp16",
              "limit": 2
            },
            {
              "url": "https://text.customer-endpoints.nimbus.octoml.ai",
              "model": "llama-2-70b-chat-int4",
              "limit": 4
            },
            {
              "url": "https://text.customer-endpoints.nimbus.octoml.ai",
              "model": "llama-2-70b-chat-fp16",
              "limit": 4
            },
          ],
      "codellama":
          [
            {
              "url": "https://text.customer-endpoints.nimbus.octoml.ai",
              "model": "codellama-7b-instruct-fp16",
              "limit": 2
            },
            {
              "url": "https://text.customer-endpoints.nimbus.octoml.ai",
              "model": "codellama-13b-instruct-fp16",
              "limit": 3
            },
            {
              "url": "https://text.customer-endpoints.nimbus.octoml.ai",
              "model": "codellama-34b-instruct-int4",
              "limit": 3
            },
            {
              "url": "https://text.customer-endpoints.nimbus.octoml.ai",
              "model": "codellama-34b-instruct-fp16",
              "limit": 3
            },

          ],
      "mistral":
          [
            {
              "url": "https://text.customer-endpoints.nimbus.octoml.ai",
              "model": "mistral-7b-instruct-fp16",
              "limit": 3
            }
          ],
    }
]

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