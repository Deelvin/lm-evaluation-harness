import json
import subprocess
import os

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'endpoints.json'), 'r') as f:
    # Загружаем данные из файла
    endpoints_data = json.load(f)

token = os.getenv("OCTOAI_TOKEN")
current_directory = os.getcwd()
path = "test_results"
prod = "Prod"

if not os.path.exists(os.path.join(current_directory, path)):
    os.makedirs(path)
if os.path.exists(os.path.join(path, "test_results.csv")):
    os.remove(os.path.join(path, "test_results.csv"))

for model_info in endpoints_data[prod]:
    model_name = model_info["model"]
    model_url = model_info["url"]

    docker_command = f"docker run -v {current_directory}/{path}:/lm_eval/test_results -e OCTOAI_TOKEN={token} daniilbarinov/lm-eval:1.0 pytest tests/smoke_accuracy_tests.py -vv --model_name {model_name} --endpoint {model_url}"

    subprocess.run(docker_command, shell=True)
