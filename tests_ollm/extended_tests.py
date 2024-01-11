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
    is_stream_type,
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
    send_request_with_timeout(url, request, headers)

    follow_up_request = send_request_get_response(url, request, headers).json()
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
        futures = [
            executor.submit(send_request_get_response, url, request, headers)
            for _ in range(num_workers)
        ]
        for future in concurrent.futures.as_completed(futures):
            responses_code_set.add(future.result().status_code)
    first_run_time = time.time() - start_time
    assert responses_code_set == {200}, "There is a problem with sending request"

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(send_request_with_timeout, url, request, headers)
            for _ in range(num_workers)
        ]

    start_time = time.time()
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(send_request_get_response, url, request, headers)
            for _ in range(num_workers)
        ]
        for future in concurrent.futures.as_completed(futures):
            responses_code_set.add(future.result().status_code)
    second_run_time = time.time() - start_time
    assert responses_code_set == {200}, "There is a problem with sending request"

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
        model_name, messages, token, endpoint, max_tokens=100, return_completion=True
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
    if model_name == "codellama-34b-instruct-fp16" and endpoint == "https://text.octoai.run":
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
        futures = [
            executor.submit(send_request_get_response, url, request, headers)
            for _ in range(num_workers)
        ]
        for future in concurrent.futures.as_completed(futures):
            responses_code_set.add(future.result().status_code)

    assert responses_code_set == {200}


@pytest.mark.parametrize("num_workers", [64, 128])
@pytest.mark.parametrize("input_tokens", [2031, 3957])
def test_send_many_large_input_content(num_workers, input_tokens, model_name, context_size, token, endpoint):
    if input_tokens >= context_size:
        pytest.skip("Input tokens are bigger than context_size")

    with open(path_to_file(f"input_context/text_about_{input_tokens}_tokens.txt"), "r") as file:
        prompt = file.read()
    request = model_data(model_name, prompt, max_tokens=1)
    url = endpoint + "/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }
    responses_code_set = set()

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(send_request_get_response, url, request, headers)
            for _ in range(num_workers)
        ]
        for future in concurrent.futures.as_completed(futures):
            responses_code_set.add(future.result().status_code)

    assert responses_code_set == {200}


@pytest.mark.parametrize("num_workers", [64, 128])
def test_send_small_and_many_big_input_contents(num_workers, model_name, context_size, token, endpoint):
    with open(path_to_file(f"input_context/text_about_496_tokens.txt"), "r") as file:
        small_prompt = file.read()

    if context_size >= 4096:
        input_tokens = 3957
    else:
        input_tokens = 2031

    with open(path_to_file(f"input_context/text_about_{input_tokens}_tokens.txt"), "r") as file:
        big_prompt = file.read()

    small_request = model_data(model_name, small_prompt, max_tokens=10)
    big_request = model_data(model_name, big_prompt, max_tokens=10)
    url = endpoint + "/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }
    responses_code_set = set()

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(send_request_get_response, url, small_request, headers)]
        futures += [
            executor.submit(send_request_get_response, url, big_request, headers)
            for _ in range(num_workers)
        ]
        for future in concurrent.futures.as_completed(futures):
            responses_code_set.add(future.result().status_code)

    assert responses_code_set == {200}


def test_send_increasing_sequence_of_contents(model_name, context_size, token, endpoint):
    input_tokens = [496, 963, 2031, 3119, 3957, 5173]
    prompts = []
    for input in input_tokens:
        if input > context_size:
            break
        with open(path_to_file(f"input_context/text_about_{input}_tokens.txt"), "r") as file:
            prompts.append(file.read()) 

    requests = []
    for i, prompt in enumerate(prompts):
        request = model_data(model_name, prompt, max_tokens=10)
        how_many = (i + 1) * 5
        requests += [request] * how_many
    num_workers = len(requests)

    url = endpoint + "/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }
    responses_code_set = set()

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(send_request_get_response, url, request, headers)
            for request in requests
        ]
        for future in concurrent.futures.as_completed(futures):
            responses_code_set.add(future.result().status_code)

    assert responses_code_set == {200}


@pytest.mark.parametrize("n", [1000, 1500, 2000, 2200, 2300, 2500])
def test_large_number_chat_completions(model_name, n, token, endpoint):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]
    completion = run_completion(
        model_name, messages, token, endpoint, n=n, return_completion=True
    )
    assert len(completion["choices"]) == n


@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_all_completions_same(model_name, n, token, endpoint):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]
    completion = run_completion(
        model_name, messages, token, endpoint, n=n, temperature=0.001, return_completion=True
    )
    content_arr = set([completion["choices"][i]["message"]["content"] for i in range(n)])

    assert len(content_arr) == 1


@pytest.mark.parametrize("num_workers", [2])
@pytest.mark.parametrize("n", [10])
@pytest.mark.xfail(reason="Currently, there is a problem with n")
def test_many_request_and_completion(model_name, num_workers, n, token, endpoint):
    message = "Create a short story about a friendship between a cat and a dog."
    request = model_data(model_name, message, max_tokens=300, n=n)
    url = endpoint + "/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }
    responses_code_set = set()
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(send_request_get_response, url, request, headers)
            for _ in range(num_workers)
        ]
        for future in concurrent.futures.as_completed(futures):
            print(future.result().json())
            responses_code_set.add(future.result().status_code)

    assert responses_code_set == {200}


@pytest.mark.parametrize("n", [2, 10, 100])
def test_stream_with_num_chat_completion(model_name, n, token, endpoint):
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Write very short."},
        {"role": "user", "content": "Describe a dog"},
    ]

    completion_stream = run_completion(
        model_name,
        messages,
        token,
        endpoint,
        n=n,
        stream=True,
        max_tokens=300,
        temperature=0.8,
        frequency_penalty=0.8,
        return_completion=True,
    )

    assert is_stream_type(completion_stream)
    stream_finish = set()
    for chunk in completion_stream:
        chunk_index = chunk.choices[0].index
        chunk_finish_reason = chunk.choices[0].finish_reason

        assert chunk_index not in stream_finish
        if chunk_finish_reason is not None:
            stream_finish.add(chunk_index)
