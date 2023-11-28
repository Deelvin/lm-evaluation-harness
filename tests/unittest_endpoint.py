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
            # elif task_name.find("truthfulqa_gen") != -1:
            #     assert i["bleurt_acc"] == "1", f"Found the wrong answer or the incorrect scoring case:\nPredicted:\n{i['logit_0']}\nTruth:\n{i['truth']}"


@pytest.mark.parametrize("headers", cfg.HEADERS)
@pytest.mark.parametrize("endpoints_data", cfg.ENDPOINTS_DATA[0]['llama'])
def test_llama_availability(endpoints_data, headers):
    data = {
        "model": endpoints_data["model"],
        "messages": [
            {
                "role": "system",
                "content": ""
            },
            {
                "role": "user",
                "content": "Who is Python author?"
            }
        ],
        "max_tokens": 10,
        "stream": False,
    }

    response = requests.post(
        endpoints_data["url"] + "/v1/chat/completions", headers=headers, json=data
    )

    assert (
        response.status_code == 200
    ), f"HTTP Status Code for {endpoints_data['model']}: {response.status_code}"

    assert response.content.strip() != "", f"Response for {endpoints_data['model']} is empty"


@pytest.mark.parametrize("headers", cfg.HEADERS)
@pytest.mark.parametrize("endpoints_data", cfg.ENDPOINTS_DATA[0]['codellama'])
def test_codellama_availability(endpoints_data, headers):
    data = {
        "model": endpoints_data["model"],
        "messages": [
            {
                "role": "system",
                "content": ""
            },
            {
                "role": "user",
                "content": "Who is Python author?"
            }
        ],
        "max_tokens": 10,
        "stream": False,
    }

    response = requests.post(
        endpoints_data["url"] + "/v1/chat/completions", headers=headers, json=data
    )

    assert (
        response.status_code == 200
    ), f"HTTP Status Code for {endpoints_data['model']}: {response.status_code}"

    assert response.content.strip() != "", f"Response for {endpoints_data['model']} is empty"


@pytest.mark.parametrize("headers", cfg.HEADERS)
@pytest.mark.parametrize("endpoints_data", cfg.ENDPOINTS_DATA[0]['mistral'])
def test_mistral_availability(endpoints_data, headers):
    data = {
        "model": endpoints_data["model"],
        "messages": [
            {
                "role": "system",
                "content": ""
            },
            {
                "role": "user",
                "content": "Who is Python author?"
            }
        ],
        "max_tokens": 10,
        "stream": False,
    }

    response = requests.post(
        endpoints_data["url"] + "/v1/chat/completions", headers=headers, json=data
    )

    assert (
        response.status_code == 200
    ), f"HTTP Status Code for {endpoints_data['model']}: {response.status_code}"

    assert response.content.strip() != "", f"Response for {endpoints_data['model']} is empty"


@pytest.mark.parametrize("task_name", cfg.TASKS[:2])
@pytest.mark.parametrize("endpoints_data", cfg.ENDPOINTS_DATA[0]['llama'])
def test_llama_output(endpoints_data, task_name):
    num_fewshot = 0
    print(task_name)
    task_names = utils.pattern_match(task_name.split(","), tasks.ALL_TASKS)
    print(task_names)
    evaluator.simple_evaluate(
        model="octoai",
        model_args="model_name="+f"'{endpoints_data['model']}'",
        tasks=task_names,
        num_fewshot=num_fewshot,
        batch_size=1,
        max_batch_size=None,
        device=None,
        no_cache=True,
        description_dict=None,
        decontamination_ngrams_path=None,
        check_integrity=False,
        write_out=True,
        limit=endpoints_data['limit'],
        output_base_path=None,
        no_shuffle=True,
    )

    check_output(endpoints_data['model'], task_names[0], num_fewshot)


@pytest.mark.parametrize("task_name", cfg.TASKS[2:4])
@pytest.mark.parametrize("endpoints_data", cfg.ENDPOINTS_DATA[0]['codellama'])
def test_codellama_output(endpoints_data, task_name):
    num_fewshot = 0
    print(task_name)
    task_names = utils.pattern_match(task_name.split(","), tasks.ALL_TASKS)
    print(task_names)
    evaluator.simple_evaluate(
        model="octoai",
        model_args="model_name="+f"'{endpoints_data['model']}'",
        tasks=task_names,
        num_fewshot=num_fewshot,
        batch_size=1,
        max_batch_size=None,
        device=None,
        no_cache=True,
        description_dict=None,
        decontamination_ngrams_path=None,
        check_integrity=False,
        write_out=True,
        limit=endpoints_data['limit'],
        output_base_path=None,
        no_shuffle=True,
    )

    check_output(endpoints_data['model'], task_names[0], num_fewshot)


@pytest.mark.parametrize("task_name", cfg.TASKS[4:6])
@pytest.mark.parametrize("endpoints_data", cfg.ENDPOINTS_DATA[0]['mistral'])
def test_mistral_output(endpoints_data, task_name):
    num_fewshot = 0
    print(task_name)
    task_names = utils.pattern_match(task_name.split(","), tasks.ALL_TASKS)
    print(task_names)
    evaluator.simple_evaluate(
        model="octoai",
        model_args="model_name="+f"'{endpoints_data['model']}'",
        tasks=task_names,
        num_fewshot=num_fewshot,
        batch_size=1,
        max_batch_size=None,
        device=None,
        no_cache=True,
        description_dict=None,
        decontamination_ngrams_path=None,
        check_integrity=False,
        write_out=True,
        limit=endpoints_data['limit'],
        output_base_path=None,
        no_shuffle=True,
    )

    check_output(endpoints_data['model'], task_names[0], num_fewshot)


# @pytest.mark.parametrize("task_name", cfg.TASKS[:2])
# @pytest.mark.parametrize("models", cfg.MODEL_ENDPOINTS[:3])
# def test_llama2_7b(models, task_name):
#     num_fewshot = 0
#     model_name = models[0]
#     print(model_name)
#     task_names = utils.pattern_match(task_name.split(","), tasks.ALL_TASKS)

#     evaluator.simple_evaluate(
#         model="octoai",
#         model_args=models[0],
#         tasks=task_names,
#         num_fewshot=num_fewshot,
#         batch_size=1,
#         max_batch_size=None,
#         device=None,
#         no_cache=True,
#         description_dict=None,
#         decontamination_ngrams_path=None,
#         check_integrity=False,
#         write_out=True,
#         output_base_path=None,
#     )

#     check_output(model_name, task_names[0], num_fewshot)


# @pytest.mark.parametrize("task_name", cfg.TASKS[2:4])
# @pytest.mark.parametrize("models", cfg.MODEL_ENDPOINTS[3:6])
# def test_llama2_13b(models, task_name):
#     num_fewshot = 0
#     model_name = models[0][12:-1]
#     task_names = utils.pattern_match(task_name.split(","), tasks.ALL_TASKS)

#     evaluator.simple_evaluate(
#         model="octoai",
#         model_args=models[0],
#         tasks=task_names,
#         num_fewshot=num_fewshot,
#         batch_size=1,
#         max_batch_size=None,
#         device=None,
#         no_cache=True,
#         description_dict=None,
#         decontamination_ngrams_path=None,
#         check_integrity=False,
#         write_out=True,
#         output_base_path=None,
#     )

#     check_output(model_name, task_names[0], num_fewshot)


# @pytest.mark.parametrize("task_name", cfg.TASKS[4:6])
# @pytest.mark.parametrize("models", cfg.MODEL_ENDPOINTS[6:9])
# def test_llama2_70b(models, task_name):
#     num_fewshot = 0
#     model_name = models[0][12:-1]
#     task_names = utils.pattern_match(task_name.split(","), tasks.ALL_TASKS)
#     print(task_name)
#     print(task_names)

#     evaluator.simple_evaluate(
#         model="octoai",
#         model_args=models[0],
#         tasks=task_names,
#         num_fewshot=num_fewshot,
#         batch_size=1,
#         max_batch_size=None,
#         device=None,
#         no_cache=True,
#         description_dict=None,
#         decontamination_ngrams_path=None,
#         check_integrity=False,
#         write_out=True,
#         output_base_path=None,
#     )

#     check_output(model_name, task_names[0], num_fewshot)


# # @pytest.mark.parametrize("cfg_tasks", cfg.TASKS)
# # @pytest.mark.parametrize("models", cfg.MODEL_ENDPOINTS)
# # def test_llama2_7b_13b(models, cfg_tasks):
# #     num_fewshot = 0
# #     model_name = models[0][12:-1]
# #     task_names = utils.pattern_match(cfg_tasks.split(","), tasks.ALL_TASKS)

# #     evaluator.simple_evaluate(
# #         model="octoai",
# #         model_args=models[0],
# #         tasks=task_names,
# #         num_fewshot=num_fewshot,
# #         batch_size=1,
# #         max_batch_size=None,
# #         device=None,
# #         no_cache=True,
# #         limit=3,
# #         description_dict=None,
# #         decontamination_ngrams_path=None,
# #         check_integrity=False,
# #         write_out=True,
# #         output_base_path=None,
# #         samples_choice=["10","40","49"]
# #     )

# #     check_output(model_name, task_names[0], num_fewshot)
