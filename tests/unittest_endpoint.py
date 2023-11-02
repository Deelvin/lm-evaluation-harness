import requests
import pytest
from lm_eval import tasks, evaluator, utils
import testdata.config as cfg
import json

@pytest.mark.parametrize("headers", cfg.HEADERS)
@pytest.mark.parametrize("models", cfg.MODEL_ENDPOINTS)
def test_endpoints_availability(models, headers):
    data = {
        "model": models[1],
        "messages": [
            {
                "role": "user",
                "content": ""
            }
        ],
        "stream": False,
        "max_tokens": 256
    }

    response = requests.post(models[1]+"/v1/chat/completions", headers=headers, json=data)

    assert response.status_code == 200, f"HTTP Status Code for {models[0]}: {response.status_code}"

    assert response.content.strip() != "", f"Response for {models[0]} is empty"
    

@pytest.mark.parametrize("cfg_tasks", cfg.TASKS)
@pytest.mark.parametrize("headers", cfg.HEADERS)
@pytest.mark.parametrize("models", cfg.MODEL_ENDPOINTS[:3])
def test_llama2_7b_indexes(models, cfg_tasks, headers):
    num_fewshot = 0

    model_name = models[0][12:-1]

    task_names = utils.pattern_match(cfg_tasks.split(","), tasks.ALL_TASKS)

    evaluator.simple_evaluate(
        model="octoai",
        model_args=models[0],
        tasks=task_names,
        num_fewshot=num_fewshot,
        batch_size=1,
        max_batch_size=None,
        device=None,
        no_cache=True,
        limit=3,
        description_dict=None,
        decontamination_ngrams_path=None,
        check_integrity=False,
        write_out=True,
        output_base_path=None,
        samples_choice=["10","25","40","41","49"]
    )

    write_out_path = f"{model_name}_{task_names[0]}_{num_fewshot}fs_write_out_info.json"
    with open(write_out_path, "r") as f:
        evaluated_output = json.load(f)
        
        assert evaluated_output is not []

        for i in evaluated_output:
            assert i["acc"] == "True", f"Found the wrong answer or the incorrect scoring case:\nPredicted:\n{i['logit_0']}\nTruth:\n{i['truth']}"


@pytest.mark.parametrize("headers", cfg.HEADERS)
@pytest.mark.parametrize("models", cfg.MODEL_ENDPOINTS[:3])
def test_llama2_7b_gsm8k_truncated(models, headers, task_name="gsm8k_truncated"):
    num_fewshot = 0

    model_name = models[0][12:-1]

    task_names = utils.pattern_match(task_name.split(","), tasks.ALL_TASKS)

    evaluator.simple_evaluate(
        model="octoai",
        model_args=models[0],
        tasks=task_names,
        num_fewshot=num_fewshot,
        batch_size=1,
        max_batch_size=None,
        device=None,
        no_cache=True,
        limit=3,
        description_dict=None,
        decontamination_ngrams_path=None,
        check_integrity=False,
        write_out=True,
        output_base_path=None,
    )

    write_out_path = f"{model_name}_{task_names[0]}_{num_fewshot}fs_write_out_info.json"
    with open(write_out_path, "r") as f:
        evaluated_output = json.load(f)
        
        assert evaluated_output is not []

        for i in evaluated_output:
            assert i["acc"] == "True", f"Found the wrong answer or the incorrect scoring case:\nPredicted:\n{i['logit_0']}\nTruth:\n{i['truth']}"


@pytest.mark.parametrize("cfg_tasks", cfg.TASKS)
@pytest.mark.parametrize("headers", cfg.HEADERS)
@pytest.mark.parametrize("models", cfg.MODEL_ENDPOINTS[3:6])
def test_llama2_13b_indexes(models, cfg_tasks, headers):
    num_fewshot = 0

    model_name = models[0][12:-1]

    task_names = utils.pattern_match(cfg_tasks.split(","), tasks.ALL_TASKS)

    evaluator.simple_evaluate(
        model="octoai",
        model_args=models[0],
        tasks=task_names,
        num_fewshot=num_fewshot,
        batch_size=1,
        max_batch_size=None,
        device=None,
        no_cache=True,
        limit=3,
        description_dict=None,
        decontamination_ngrams_path=None,
        check_integrity=False,
        write_out=True,
        output_base_path=None,
        samples_choice=["0","7","9","28","32"]
    )

    write_out_path = f"{model_name}_{task_names[0]}_{num_fewshot}fs_write_out_info.json"
    with open(write_out_path, "r") as f:
        evaluated_output = json.load(f)
        
        assert evaluated_output is not []

        for i in evaluated_output:
            assert i["acc"] == "True", f"Found the wrong answer or the incorrect scoring case:\nPredicted:\n{i['logit_0']}\nTruth:\n{i['truth']}"
    

@pytest.mark.parametrize("cfg_tasks", cfg.TASKS)
@pytest.mark.parametrize("headers", cfg.HEADERS)
@pytest.mark.parametrize("models", cfg.MODEL_ENDPOINTS)
def test_llama2_7b_13b(models, cfg_tasks, headers):
    num_fewshot = 0

    model_name = models[0][12:-1]

    task_names = utils.pattern_match(cfg_tasks.split(","), tasks.ALL_TASKS)

    evaluator.simple_evaluate(
        model="octoai",
        model_args=models[0],
        tasks=task_names,
        num_fewshot=num_fewshot,
        batch_size=1,
        max_batch_size=None,
        device=None,
        no_cache=True,
        limit=3,
        description_dict=None,
        decontamination_ngrams_path=None,
        check_integrity=False,
        write_out=True,
        output_base_path=None,
        samples_choice=["10","40","49"]
    )

    write_out_path = f"{model_name}_{task_names[0]}_{num_fewshot}fs_write_out_info.json"
    with open(write_out_path, "r") as f:
        evaluated_output = json.load(f)
        
        assert evaluated_output is not []

        for i in evaluated_output:
            assert i["acc"] == "True", f"Found the wrong answer or the incorrect scoring case:\nPredicted:\n{i['logit_0']}\nTruth:\n{i['truth']}"