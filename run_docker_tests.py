import json
import subprocess
import os

token = os.getenv("OCTOAI_TOKEN")
current_directory = os.getcwd()
path = "test_results"
# Читаем endpoints.json
with open("endpoints.json", "r") as f:
    endpoints_data = json.load(f)

if not os.path.exists(os.path.join(current_directory, path)):
    os.makedirs(path)
else:
    os.remove(os.path.join(path, "test_results.csv"))

# Проходим по каждой модели
for model_info in endpoints_data["Prod"]:
    model_name = model_info["model"]
    # Строим команду docker run
    docker_command = f"docker run -v {current_directory}:/lm_eval/test_results -e OCTOAI_TOKEN={token} lm-eval pytest tests/unittest_endpoint.py -vv --model_name {model_name} --endpoint Prod"

    # Выполняем команду
    subprocess.run(docker_command, shell=True)
