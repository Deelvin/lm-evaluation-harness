import requests
import pytest
from lm_eval import tasks, evaluator, utils
import testdata.config as cfg
import json

@pytest.mark.smoke
@pytest.mark.parametrize("cfg_tasks", cfg.TASKS)
@pytest.mark.parametrize("headers", cfg.HEADERS)
@pytest.mark.parametrize("models", cfg.MODEL_ARGS)
def test_model_endpoint(models, cfg_tasks, headers):
    num_fewshot = 0

    model_name = models[12:-1]

    task_names = utils.pattern_match(cfg_tasks.split(","), tasks.ALL_TASKS)

    evaluator.simple_evaluate(
        model="octoai",
        model_args=models,
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
    print("OUTPUT PATH2", write_out_path)
    with open(write_out_path, "r") as f:
        evaluated_output = json.load(f)
        
        assert evaluated_output is not []

        for i in evaluated_output:
            assert i["acc"] == "True", f"Found the wrong answer or the incorrect scoring case:\nPredicted:\n{i['logit_0']}\nTruth:\n{i['truth']}"

    # a = [True, False, True]
    # res = a.all()

    # data = {
    #     "model": model_endpoint,
    #     "messages": [
    #         {
    #             "role": "user",
    #             "content": "" # need to fill before use inference
    #         }
    #     ],
    #     "stream": False,
    #     "max_tokens": 256
    # }

    
    # data["messages"][0]["content"] = prompt
    # response = requests.post(model_endpoint+"/v1/chat/completions", headers=headers, json=data)
    # print(response)
    # assert response.status_code == 200, f"HTTP Status Code for {model_endpoint}: {response.status_code}"

    # assert response.content.strip() != "", f"Response for {model_endpoint} is empty"
    # response_data = response.json()
    # print(response_data)
    # print(response.content)

    
    # assert "content" in response_data['choices'][0]["message"]
    
    # model_output = response_data['choices'][0]["message"]["content"]
    # assert re.search(expected_pattern, model_output)
