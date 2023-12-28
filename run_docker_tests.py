import json
import subprocess
import os

endpoints_data = {
    "Dev": [
        {
            "url": "https://text.customer-endpoints.nimbus.octoml.ai",
            "model": "codellama-7b-instruct-fp16",
        },
        {
            "url": "https://text.customer-endpoints.nimbus.octoml.ai",
            "model": "codellama-13b-instruct-fp16",
        },
        {
            "url": "https://text.customer-endpoints.nimbus.octoml.ai",
            "model": "codellama-34b-instruct-int4",
        },
        {
            "url": "https://text.customer-endpoints.nimbus.octoml.ai",
            "model": "codellama-34b-instruct-fp16",
        },
        {
            "url": "https://text.customer-endpoints.nimbus.octoml.ai",
            "model": "llama-2-13b-chat-fp16",
        },
        {
            "url": "https://text.customer-endpoints.nimbus.octoml.ai",
            "model": "llama-2-70b-chat-int4",
        },
        {
            "url": "https://text.customer-endpoints.nimbus.octoml.ai",
            "model": "llama-2-70b-chat-fp16",
        },
        {
            "url": "https://text.customer-endpoints.nimbus.octoml.ai",
            "model": "mistral-7b-instruct-fp16",
        },
        {
            "url": "https://text.customer-endpoints.nimbus.octoml.ai",
            "model": "mixstral-8x7b-instruct-fp16",
        },
    ],
    "Prod": [
        {"url": "https://text.octoai.run", "model": "codellama-7b-instruct-fp16"},
        {"url": "https://text.octoai.run", "model": "codellama-13b-instruct-fp16"},
        {"url": "https://text.octoai.run", "model": "codellama-34b-instruct-int4"},
        {"url": "https://text.octoai.run", "model": "codellama-34b-instruct-fp16"},
        {"url": "https://text.octoai.run", "model": "llama-2-13b-chat-fp16"},
        {"url": "https://text.octoai.run", "model": "llama-2-70b-chat-int4"},
        {"url": "https://text.octoai.run", "model": "llama-2-70b-chat-fp16"},
        {"url": "https://text.octoai.run", "model": "mistral-7b-instruct-fp16"},
    ],
}

token = os.getenv("OCTOAI_TOKEN")
current_directory = os.getcwd()
path = "test_results"

if not os.path.exists(os.path.join(current_directory, path)):
    os.makedirs(path)
else:
    os.remove(os.path.join(path, "test_results.csv"))

for model_info in endpoints_data["Prod"]:
    model_name = model_info["model"]

    docker_command = f"docker run -v {current_directory}:/lm_eval/test_results -e OCTOAI_TOKEN={token} lm-eval pytest tests/unittest_endpoint.py -vv --model_name {model_name} --endpoint Prod"

    subprocess.run(docker_command, shell=True)
