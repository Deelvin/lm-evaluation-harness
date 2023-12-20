import requests
import pytest
from lm_eval import tasks, evaluator, utils
import config as cfg
import json


def check_output(model_name, task_name, num_fewshot):
    write_out_path = f"{model_name}_{task_name}_{num_fewshot}fs_write_out_info.json"
    with open(write_out_path, "r") as f:
        evaluated_output = json.load(f)

        assert evaluated_output is not []

        for i in evaluated_output:
            if task_name.find("gsm8k") != -1:
                assert (
                    i["acc"] == "True"
                ), f"Found the wrong answer or the incorrect scoring case:\nPredicted:\n{i['logit_0']}\nTruth:\n{i['truth']}"
            elif task_name.find("triviaqa") != -1:
                assert (
                    i["em"] == "1.0"
                ), f"Found the wrong answer or the incorrect scoring case:\nPredicted:\n{i['logit_0']}\nTruth:\n{i['truth']}"


@pytest.fixture
def model_name(request):
    return request.config.getoption("--model_name")


@pytest.fixture
def endpoint(request):
    return request.config.getoption("--endpoint")


@pytest.fixture
def token(request):
    return os.getenv("OCTOAI_TOKEN")


def test_endpoint_availability(model_name, endpoint, token):
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}

    data = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": "Who is Python author?"},
        ],
        "max_tokens": 10,
        "stream": False,
    }

    response = requests.post(
        endpoint + "/v1/chat/completions", headers=headers, json=data
    )

    assert (
        response.status_code == 200
    ), f"HTTP Status Code for {model_name}: {response.status_code}"

    assert response.content.strip() != "", f"Response for {model_name} is empty"


def test_endpoint_gsm8k(model_name, endpoint, token):
    num_fewshot = 0
    task_name = ["gsm8k_truncated_" + model_name.split("-")[0]]

    evaluator.simple_evaluate(
        model="octoai",
        model_args=f"model_name='{model_name}',prod=True,token='{token}'",
        tasks=task_name,
        num_fewshot=num_fewshot,
        batch_size=1,
        max_batch_size=None,
        device=None,
        no_cache=True,
        description_dict=None,
        decontamination_ngrams_path=None,
        check_integrity=False,
        write_out=True,
        limit=cfg.ENDPOINTS_DATA["gsm8k"][model_name],
        output_base_path=None,
        no_shuffle=True,
    )

    check_output(model_name, task_name[0], num_fewshot)


def test_endpoint_triviaqa(model_name, endpoint, token):
    num_fewshot = 0
    task_name = ["triviaqa_truncated_" + model_name.split("-")[0]]

    evaluator.simple_evaluate(
        model="octoai",
        model_args=f"model_name='{model_name}',prod=True,token='{token}'",
        tasks=task_name,
        num_fewshot=num_fewshot,
        batch_size=1,
        max_batch_size=None,
        device=None,
        no_cache=True,
        description_dict=None,
        decontamination_ngrams_path=None,
        check_integrity=False,
        write_out=True,
        limit=cfg.ENDPOINTS_DATA["triviaqa"][model_name],
        output_base_path=None,
        no_shuffle=True,
    )

    check_output(model_name, task_name[0], num_fewshot)
