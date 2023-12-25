import os
import time

import pytest


import numpy as np
from scipy.spatial import distance
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

import openai

from utils import (
    path_to_file,
    run_completion,
    send_request_with_timeout,
    send_request_get_response,
)

# For compatibility with OpenAI versions before v1.0
# https://github.com/openai/openai-python/pull/677.
OPENAI_VER_MAJ = int(openai.__version__.split(".")[0])

if OPENAI_VER_MAJ >= 1:
    from openai import APIError
else:
    from openai.error import APIError

from sentence_transformers import SentenceTransformer


# Define the model_name fixture
@pytest.fixture
def model_name(request):
    return request.config.getoption("--model_name")


@pytest.fixture
def token():
    return os.environ["OCTOAI_TOKEN"]


@pytest.fixture
def endpoint(request):
    return request.config.getoption("--endpoint")


@pytest.fixture
def context_size(request):
    return request.config.getoption("--context_size", default=4096)


def test_response(model_name, token, endpoint):
    prompt = "Hello!"
    assert run_completion(model_name, prompt, token, endpoint, chat=False) == 200

    model_name += "_dummy_check"
    assert run_completion(model_name, prompt, token, endpoint, chat=False) != 200


@pytest.mark.parametrize("max_tokens", [10, 100, 300, 500, 1024])
def test_max_tokens(model_name, max_tokens, token, endpoint):
    prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request. Write a really long blog about Seattle."
    completion = run_completion(
        model_name,
        prompt,
        token,
        endpoint, 
        chat=False,
        max_tokens=max_tokens,
        return_completion=True,
    )
    assert 0 < completion["usage"]["completion_tokens"] <= max_tokens
    assert len(completion["choices"][0]["text"]) > 0


@pytest.mark.skip(
    reason="Due to Internal Server Error (500) hides expected invalid_request_error (400)"
)
def test_incorrect_max_tokens(model_name, context_size, token, endpoint):
    prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request. Write a really long blog about Seattle."

    assert run_completion(model_name, prompt, token, endpoint, max_tokens=-1, chat=False) == 400
    assert (
        run_completion(model_name, prompt, token, endpoint, max_tokens=context_size * 2, chat=False)
        == 400
    )
    completion = run_completion(
        model_name,
        prompt,
        token,
        endpoint,
        max_tokens=context_size,
        return_completion=True,
        chat=False
    )
    assert 0 < completion["usage"]["completion_tokens"] <= context_size
    assert len(completion["choices"][0]["text"]) > 0


@pytest.mark.skip(reason="Need to validate distance measurement approach")
def test_valid_temperature(model_name, token, endpoint):
    """The higher the temperature, the further the distance from the expected."""
    prompt = "You are a helpful assistant. Write a blog about Seattle"
    max_tokens = 784
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    goldev_embed = np.load(path_to_file("golden_temp_0.npy"))

    distances = []
    for temperature in [0.0, 1.0, 2.0]:
        completion = run_completion(
            model_name,
            prompt,
            token,
            endpoint,
            max_tokens=max_tokens,
            temperature=temperature,
            return_completion=True,
            chat=False
        )
        curr_embeddings = model.encode(completion["choices"][0]["text"])
        distances.append(distance.cosine(curr_embeddings, goldev_embed))

    assert distances == sorted(distances)


@pytest.mark.parametrize("temperature", [-0.1, 2.1])
def test_temperature_outside_limit(model_name, temperature, token, endpoint):
    """Invalid temperatures should produce an error.

    Temperature is allowed to range from 0 to 2.0. Outside of this
    range, an error should be returned in the completion.
    """
    prompt = "You are a helpful assistant. Write a blog about Seattle"

    with pytest.raises(APIError):
        completion = run_completion(
            model_name,
            prompt,
            token,
            endpoint,
            temperature=temperature,
            return_completion=True,
            chat=False
        )


@pytest.mark.skip(reason="Need to validate distance measurement approach")
def test_top_p(model_name, token, endpoint):
    prompt = "You are a helpful assistant. Write a blog about Seattle"
    max_tokens = 784
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    goldev_embed = np.load(path_to_file("golden_top_p_0.npy"))
    completion = run_completion(
        model_name,
        prompt,
        token,
        endpoint,
        max_tokens=max_tokens,
        top_p=0.00001,
        return_completion=True,
        chat=False
    )
    curr_embeddings = model.encode(completion["choices"][0]["text"])
    prev_dist = distance.cosine(curr_embeddings, goldev_embed)

    for top_p in [0.2, 1.0]:
        completion = run_completion(
            model_name,
            prompt,
            token,
            endpoint,
            max_tokens=max_tokens,
            top_p=top_p,
            return_completion=True,
            chat=False
        )
        curr_embeddings = model.encode(completion["choices"][0]["text"])
        cur_distance = distance.cosine(curr_embeddings, goldev_embed)
        assert prev_dist <= cur_distance
        prev_dist = cur_distance


@pytest.mark.parametrize("top_p", [-0.1, 1.1])
def test_top_p_outside_limit(model_name, top_p, token, endpoint):
    prompt = "You are a helpful assistant. Write a blog about Seattle"

    assert run_completion(model_name, prompt, token, endpoint, top_p=top_p, chat=False) == 400


@pytest.mark.parametrize("n", [1, 5, 10])
def test_number_chat_completions(model_name, n, token, endpoint):
    if n > 1:
        pytest.skip("Multiple outputs is not supported yet")
    prompt = "You are a helpful assistant. Hello!"
    completion = run_completion(
        model_name, prompt, token, endpoint, n=n, return_completion=True, chat=False
    )
    assert len(completion["choices"]) == n


@pytest.mark.skip(
    reason="Need upstream with OpenAI approach (return completion without stop token)"
)
@pytest.mark.parametrize("stop", [["tomato", "tomatoes"], [".", "!"]])
def test_stop(model_name, stop, token, endpoint):
    prompt = "How to cook tomato paste?"
    completion = run_completion(
        model_name,
        prompt,
        token,
        endpoint,
        max_tokens=300,
        stop=stop,
        return_completion=True,
        chat=False
    )
    for seq in stop:
        assert (seq not in completion["choices"][0]["text"]) and completion["choices"][
            0
        ]["finish_reason"] == "stop"

    # These test assumes that stop token return with completion
    # assert completion["choices"][0]["finish_reason"] == "stop"
    # text = completion["choices"][0]["text"]
    # if "." in stop:
    #     assert text[-1] in stop
    #     for seq in stop:
    #         assert seq not in text[:-1]
    # else:
    #     words = text.split()
    #     assert words[-1] in stop
    #     for seq in stop:
    #         assert seq not in words[:-1]


def test_model_name(model_name, token, endpoint):
    prompt = "You are a helpful assistant. Hello!"
    completion = run_completion(
        model_name, prompt, token, endpoint, return_completion=True, chat=False
    )
    assert completion["model"] == model_name


def test_choices_exist(model_name, token, endpoint):
    prompt = "You are a helpful assistant. Hello!"
    completion = run_completion(
        model_name, prompt, token, endpoint, return_completion=True, chat=False
    )
    assert "choices" in completion.keys()
    assert "index" in completion["choices"][0].keys()
    assert "finish_reason" in completion["choices"][0].keys()
    assert "text" in completion["choices"][0].keys()


def test_usage(model_name, token, endpoint):
    prompt = "You are a helpful assistant. Hello!"
    completion = run_completion(
        model_name, prompt, token, endpoint, return_completion=True, chat=False
    )
    assert "usage" in completion.keys()
    assert "prompt_tokens" in completion["usage"].keys()
    assert "total_tokens" in completion["usage"].keys()
    assert "completion_tokens" in completion["usage"].keys()


def test_id_completion(model_name, token, endpoint):
    prompt = "You are a helpful assistant. Hello!"
    first_completion = run_completion(
        model_name, prompt, token, endpoint, return_completion=True, chat=False
    )
    second_completion = run_completion(
        model_name, prompt, token, endpoint, return_completion=True, chat=False
    )
    assert first_completion["id"] != second_completion["id"]


def test_created_time(model_name, token, endpoint):
    prompt = "You are a helpful assistant. Hello!"
    # The "created" timestamp is only provided at 1-second
    # granularity, so we shouldn't compare with a finer granularity.
    st_time = int(time.time())
    completion = run_completion(
        model_name, prompt, token, endpoint, return_completion=True, chat=False
    )
    end_time = int(time.time())
    assert st_time <= completion["created"] <= end_time


@pytest.mark.parametrize(
    "prompt",
    [
        [1, "123"],
        {"text": "How are you?"},
        ("Hello!", "You are a helpful assistant"),
        10,
        10.5,
        True,
    ],
)
def test_incorrect_content(model_name, prompt, token, endpoint):
    assert run_completion(model_name, prompt, token, endpoint, chat=False) == 400


@pytest.mark.parametrize("prompt", ["Hi!", None])
def test_content(model_name, prompt, token, endpoint):
    assert run_completion(model_name, prompt, token, endpoint, chat=False) == 200


def test_user_authentication(model_name, token, endpoint):
    prompt = "Tell a story about a cat"
    assert run_completion(model_name, prompt, "invalid", endpoint, chat=False) == 401


def test_cancel_and_follow_up_requests(model_name, token, endpoint):
    data = {
        "model": model_name,
        "prompt": "Create a big story about a friendship between a cat and a dog.",
        "max_tokens": 500,
        "n": 1,
        "stream": False,
        "stop": None,
        "temperature": 0.8,
        "top_p": 1.0,
        "presence_penalty": 0,
        "return_completion": False,
    }
    url = endpoint + "/v1/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }
    send_request_with_timeout(url, data, headers)

    follow_up_request = send_request_get_response(url, data, headers).json()
    assert "created" in follow_up_request


def test_canceling_requests(model_name, token, endpoint):
    data = {
        "model": model_name,
        "prompt": "Create a big story about a friendship between a cat and a dog.",
        "max_tokens": 1000,
        "n": 1,
        "stream": False,
        "stop": None,
        "temperature": 0.0,
        "top_p": 1.0,
        "presence_penalty": 0,
        "return_completion": False,
    }
    url = endpoint + "/v1/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }
    num_workers = 8
    responses_code_set = set()

    start_time = time.time()
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(send_request_get_response, url, data, headers)
            for _ in range(num_workers)
        ]
        for future in concurrent.futures.as_completed(futures):
            responses_code_set.add(future.result().status_code)
    first_run_time = time.time() - start_time
    assert responses_code_set == {200}, f"There is a problem with sending request"

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(send_request_with_timeout, url, data, headers)
            for _ in range(num_workers)
        ]

    start_time = time.time()
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(send_request_get_response, url, data, headers)
            for _ in range(num_workers)
        ]
        for future in concurrent.futures.as_completed(futures):
            responses_code_set.add(future.result().status_code)
    second_run_time = time.time() - start_time
    assert responses_code_set == {200}, f"There is a problem with sending request"

    threshold = 5
    assert abs(second_run_time - first_run_time) < threshold


@pytest.mark.parametrize("temperature", [0.0, 0.5, 0.7, 1.0, 1.5])
def test_same_completion_len(temperature, model_name, token, endpoint):
    prompt = "Hello, how can you help me? Answer short."
    tokens_arr = []
    mean = 0
    trials = 4
    for _ in range(trials):
        completion = run_completion(
            model_name,
            prompt,
            token,
            endpoint,
            temperature=temperature,
            top_p=1.0,
            return_completion=True,
            chat=False
        )
        mean += completion["usage"]["completion_tokens"]
        tokens_arr.append(completion["usage"]["completion_tokens"])

    mean /= trials
    threshold = 10
    assert all([abs(tokens_arr[i] - mean) <= threshold for i in range(trials)])


@pytest.mark.parametrize("input_tokens", [496, 963, 2031, 3119, 3957, 5173])
def test_large_input_content(input_tokens, model_name, context_size, token, endpoint):
    with open(
        path_to_file(f"input_context/text_about_{input_tokens}_tokens.txt"), "r"
    ) as file:
        prompt = file.read()
    max_tokens = 200
    if model_name == "codellama-34b-instruct-fp16":
        context_size = 16384

    if (input_tokens + max_tokens) < context_size:
        assert (
            run_completion(model_name, prompt, token, endpoint, max_tokens=max_tokens, chat=False)
            == 200
        )
    else:
        assert (
            run_completion(model_name, prompt, token, endpoint, max_tokens=max_tokens, chat=False)
            == 400
        )


def test_send_many_request(model_name, token, endpoint):
    data = {
        "model": model_name,
        "prompt": "Create a big story about a friendship between a cat and a dog.",
        "max_tokens": 300,
        "n": 1,
        "stream": False,
        "stop": None,
        "temperature": 0.0,
        "top_p": 1.0,
        "presence_penalty": 0,
        "return_completion": False,
    }

    url = endpoint + "/v1/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }
    responses_code_set = set()
    num_workers = 64

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(send_request_get_response, url, data, headers)
            for _ in range(num_workers)
        ]
        for future in concurrent.futures.as_completed(futures):
            responses_code_set.add(future.result().status_code)

    assert responses_code_set == {200}
