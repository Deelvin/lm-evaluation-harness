import subprocess
import os
import tests.testdata.config as cfg

model_names = [
    "llama-2-13b-chat-fp16",
    "llama-2-70b-chat-int4",
    "llama-2-70b-chat-fp16",
    "codellama-7b-instruct-fp16",
    "codellama-13b-instruct-fp16",
    "codellama-34b-instruct-int4",
    "codellama-34b-instruct-fp16",
    "mistral-7b-instruct-fp16",
]
token = os.getenv("OCTOAI_TOKEN")
endpoint = "https://text.octoai.run"  # prod

for model_name in model_names:
    print(
        [
            "pytest",
            "unittest_endpoint.py",
            f"--model_name={model_name}",
            f"--token={token}",
            f"--endpoint={endpoint}",
        ]
    )
    subprocess.call(
        [
            "pytest",
            "unittest_endpoint.py",
            f"--model_name={model_name}",
            f"--token={token}",
            f"--endpoint={endpoint}",
            "-vv",
        ]
    )
