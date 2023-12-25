from concurrent.futures import ThreadPoolExecutor
import os
import time
import concurrent.futures

import requests
import pytest

from utils import (
    run_completion,
    send_request_get_response,
    send_request_with_timeout,
    path_to_file,
    model_data,
)


@pytest.fixture(name="model_name")
def fixture_model_name(request):
    return request.config.getoption("--model_name")


@pytest.fixture(name="token")
def fixture_token():
    return os.environ["OCTOAI_TOKEN"]


@pytest.fixture(name="endpoint")
def fixture_endpoint(request):
    return request.config.getoption("--endpoint")


@pytest.fixture(name="context_size")
def fixture_context_size(request):
    return request.config.getoption("--context_size", default=4096)


@pytest.mark.scalability
def test_cancel_and_follow_up_requests(model_name, token, endpoint):
    message = "Create a big story about a friendship between a cat and a dog."
    request = model_data(model_name, message, max_tokens = 500)
    url = endpoint + "/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }
    try:
        requests.post(url, json=request, headers=headers, timeout=1)
    except requests.exceptions.Timeout:
        print("Timeout of request")

    follow_up_request = requests.post(url, json=request, headers=headers).json()
    assert "created" in follow_up_request


@pytest.mark.scalability
def test_canceling_requests(model_name, token, endpoint):
    message = "Create a big story about a friendship between a cat and a dog."
    request = model_data(model_name, message, max_tokens=1000)
    url = endpoint + "/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }
    num_workers = 8
    responses_code_set = set()

    start_time = time.time()
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(send_request_get_response, url, request, headers) for _ in range(num_workers)]
        for future in concurrent.futures.as_completed(futures):
            responses_code_set.add(future.result().status_code)
    first_run_time = time.time() - start_time
    assert (responses_code_set == {200}), f"There is a problem with sending request"
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(send_request_with_timeout, url, request, headers) for _ in range(num_workers)]

    start_time = time.time()
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(send_request_get_response, url, request, headers) for _ in range(num_workers)]
        for future in concurrent.futures.as_completed(futures):
            responses_code_set.add(future.result().status_code)
    second_run_time = time.time() - start_time
    assert (responses_code_set == {200}), f"There is a problem with sending request"

    print(first_run_time, second_run_time)
    threshold = 5
    assert abs(second_run_time - first_run_time) < threshold


@pytest.mark.input_parameter
@pytest.mark.parametrize("temperature", [0.0, 0.5, 0.7, 1.0, 1.5])
def test_same_completion_len(temperature, model_name, token, endpoint):
    messages = [{"role": "user", "content": "Hello, how can you help me? Answer short."}]
    tokens_arr = []
    mean = 0
    trials = 4
    for _ in range(trials):
        completion = run_completion(
            model_name,
            messages,
            token,
            endpoint,
            temperature=temperature,
            top_p=1.0,
            return_completion=True,
        )
        mean += completion["usage"]["completion_tokens"]
        tokens_arr.append(completion["usage"]["completion_tokens"])

    mean /= trials
    threshold = 10
    assert all(abs(tokens_arr[i] - mean) <= threshold for i in range(trials))


@pytest.mark.input_parameter
def test_multiple_messages(model_name, token, endpoint):
    messages = [
        {
            "role": "user",
            "content": "What is the capital of France?",
        },
        {
            "role": "assistant",
            "content": "Paris",
        },
        {
            "role": "user",
            "content": "2 + 2 =",
        },
    ]

    completion = run_completion(
        model_name, messages, token, endpoint, max_tokens=20, return_completion=True
    )
    assert "4" in completion["choices"][0]["message"]["content"]


@pytest.mark.input_parameter
@pytest.mark.parametrize("input_tokens", [496, 963, 2031, 3119, 3957, 5173])
def test_large_input_content(input_tokens, model_name, context_size, token, endpoint):
    with open(
        path_to_file(f"input_context/text_about_{input_tokens}_tokens.txt"), "r", encoding="utf-8"
    ) as file:
        prompt = file.read()
    messages = [{"role": "user", "content": prompt}]
    max_tokens = 200
    if model_name == "codellama-34b-instruct-fp16":
        context_size = 16384

    if (input_tokens + max_tokens) < context_size:
        assert (
            run_completion(model_name, messages, token, endpoint, max_tokens=max_tokens) == 200
        )
    else:
        assert (
            run_completion(model_name, messages, token, endpoint, max_tokens=max_tokens) == 400
        )


@pytest.mark.scalability
@pytest.mark.parametrize("num_workers", [64, 128])
def test_send_many_request(num_workers, model_name, token, endpoint):
    message = "Create a short story about a friendship between a cat and a dog."
    request = model_data(model_name, message, max_tokens=300)
    url = endpoint + "/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }
    responses_code_set = set()

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(send_request_get_response, url, request, headers) for _ in range(num_workers)]
        for future in concurrent.futures.as_completed(futures):
            responses_code_set.add(future.result().status_code)

    assert responses_code_set == {200}
