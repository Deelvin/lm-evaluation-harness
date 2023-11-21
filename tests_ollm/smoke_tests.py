import os
import types
import time

import requests
import pytest


import numpy as np
from scipy.spatial import distance
from concurrent.futures import ThreadPoolExecutor

import openai

try:
    from openai import APIError, AuthenticationError, APIConnectionError
except ImportError:
    # For compatibility with OpenAI versions before v1.0
    # https://github.com/openai/openai-python/pull/677.
    from openai.error import APIError, AuthenticationError, APIConnectionError
from sentence_transformers import SentenceTransformer

from openai.openai_object import OpenAIObject


api_key = os.environ["OCTOAI_TOKEN"]
openai.api_key = api_key
openai.api_base = os.environ["ENDPOINT"] + "/v1"


# Define the model_name fixture
@pytest.fixture
def model_name(request):
    return request.config.getoption("--model_name", default="llama-2-7b-chat")


@pytest.fixture
def context_size(request):
    return request.config.getoption("--context_size", default=2048)


def run_chat_completion(
    model_name,
    messages,
    max_tokens=300,
    n=1,
    stream=False,
    stop=None,
    temperature=0.8,
    top_p=1.0,
    frequency_penalty=0,
    presence_penalty=0,
    return_completion=False,
):
    http_response = 200

    try:
        completion = openai.ChatCompletion.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
            stream=stream,
            n=n,
            stop=stop,
            top_p=top_p,
            temperature=temperature,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )

        if return_completion:
            return completion
    except (APIError, AuthenticationError, APIConnectionError) as e:
        if return_completion:
            raise
        print(e.user_message)
        http_response = e.http_status

    return http_response


def test_response(model_name):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]
    assert run_chat_completion(model_name, messages) == 200

    model_name += "_dummy_check"
    assert run_chat_completion(model_name, messages) != 200


def test_incorrect_role(model_name):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "How are you!"},
    ]

    assert run_chat_completion(model_name, messages) == 200

    messages.append({"role": "dummy_role", "content": "dummy_content"})
    assert run_chat_completion(model_name, messages) != 200

    messages.pop()
    assert run_chat_completion(model_name, messages) == 200


@pytest.mark.parametrize("max_tokens", [1.5, 10, 100, 300, 500, 1024])
def test_max_tokens(model_name, context_size, max_tokens):
    messages = [
        {
            "role": "system",
            "content": "Below is an instruction that describes a task. Write a response that appropriately completes the request.",
        },
        {"role": "user", "content": "Write a really long blog about Seattle."},
    ]
    completion = run_chat_completion(
        model_name, messages, max_tokens, return_completion=True
    )
    assert 0 < completion["usage"]["completion_tokens"] <= max_tokens
    assert len(completion["choices"][0]["message"]["content"]) > 0


def test_incorrect_max_tokens(model_name, context_size):
    messages = [
        {
            "role": "system",
            "content": "Below is an instruction that describes a task. Write a response that appropriately completes the request.",
        },
        {"role": "user", "content": "Write a really long blog about Seattle."},
    ]
    assert run_chat_completion(model_name, messages, max_tokens=-1) == 400
    assert run_chat_completion(model_name, messages, max_tokens=context_size * 2) == 400
    completion = run_chat_completion(
        model_name, messages, max_tokens=context_size, return_completion=True
    )
    assert 0 < completion["usage"]["completion_tokens"] <= context_size
    assert len(completion["choices"][0]["message"]["content"]) > 0


def test_valid_temperature(model_name):
    """The higher the temperature, the further the distance from the expected."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a blog about Seattle"},
    ]
    max_tokens = 784
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    goldev_embed = np.load("golden_temp_0.npy")

    distances = []
    for temperature in [0.0, 1.0, 2.0]:
        completion = run_chat_completion(
            model_name,
            messages,
            max_tokens,
            temperature=temperature,
            return_completion=True,
        )
        curr_embeddings = model.encode(completion["choices"][0]["message"]["content"])
        distances.append(distance.cosine(curr_embeddings, goldev_embed))

    assert distances == sorted(distances)


def test_lower_temperature_limit(model_name):
    """Invalid temperatures should produce an error.

    Temperature is allowed to range from 0 to 2.0.  Outside of this
    range, an error should be thrown.
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a blog about Seattle"},
    ]

    with pytest.raises(APIError):
        completion = run_chat_completion(
            model_name, messages, temperature=-0.1, return_completion=True
        )


def test_upper_temperature_limit(model_name):
    """Invalid temperatures should produce an error.

    Temperature is allowed to range from 0 to 2.0.  Outside of this
    range, an error should be thrown.
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a blog about Seattle"},
    ]

    with pytest.raises(APIError):
        completion = run_chat_completion(
            model_name, messages, temperature=2.1, return_completion=True
        )


def test_top_p(model_name):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a blog about Seattle"},
    ]
    max_tokens = 784
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    goldev_embed = np.load("golden_top_p_0.npy")
    completion = run_chat_completion(
        model_name, messages, max_tokens, top_p=0.00001, return_completion=True
    )
    curr_embeddings = model.encode(completion["choices"][0]["message"]["content"])
    prev_dist = distance.cosine(curr_embeddings, goldev_embed)

    for top_p in [0.2, 1.0]:
        completion = run_chat_completion(
            model_name, messages, max_tokens, top_p=top_p, return_completion=True
        )
        curr_embeddings = model.encode(completion["choices"][0]["message"]["content"])
        cur_distance = distance.cosine(curr_embeddings, goldev_embed)
        assert prev_dist <= cur_distance
        prev_dist = cur_distance

    assert run_chat_completion(model_name, messages, top_p=-0.1) == 400
    assert run_chat_completion(model_name, messages, top_p=1.1) == 400


@pytest.mark.parametrize("n", [1, 5, 10])
def test_number_chat_completions(model_name, n):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]
    completion = run_chat_completion(model_name, messages, n=n, return_completion=True)
    assert len(completion["choices"]) == n


def test_stream(model_name):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a blog about Seattle"},
    ]
    completion_stream = run_chat_completion(
        model_name, messages, stream=True, temperature=0.0, return_completion=True
    )
    assert isinstance(completion_stream, types.GeneratorType)
    stream_str = ""
    for chunk in completion_stream:
        if chunk["choices"][0]["delta"]["role"] != "assistant":
            stream_str += chunk["choices"][0]["delta"]["content"]
    completion = run_chat_completion(
        model_name, messages, stream=False, temperature=0.0, return_completion=True
    )
    assert stream_str == completion["choices"][0]["message"]["content"]


@pytest.mark.parametrize("stop", [["tomato", "tomatoes"], [".", "!"]])
def test_stop(model_name, stop):
    messages = [{"role": "user", "content": "How to cook tomato paste?"}]
    completion = run_chat_completion(
        model_name, messages, stop=stop, return_completion=True
    )
    for seq in stop:
        assert (
            seq not in completion["choices"][0]["message"]["content"]
        ) and completion["choices"][0]["finish_reason"] == "stop"


def test_frequency_penalty(model_name):
    messages = [
        {
            "role": "system",
            "content": "You are content maker. Write the response in formal style that appropriately completes the request",
        },
        {
            "role": "user",
            "content": "Write a 1000-word article about the development of computer science",
        },
    ]
    max_tokens = 1500
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    initial_completion = run_chat_completion(
        model_name, messages, return_completion=True
    )
    initial_embeddings = model.encode(
        initial_completion["choices"][0]["message"]["content"]
    )
    messages.append(
        {
            "role": "assistant",
            "content": initial_completion["choices"][0]["message"]["content"],
        }
    )
    messages.append(
        {
            "role": "user",
            "content": "Write a 1000-word article about the development of computer science",
        }
    )
    first_completion = run_chat_completion(
        model_name,
        messages,
        max_tokens=max_tokens,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        temperature=0.0,
        return_completion=True,
    )
    second_completion = run_chat_completion(
        model_name,
        messages,
        max_tokens=max_tokens,
        frequency_penalty=2.0,
        presence_penalty=0.0,
        temperature=0.0,
        return_completion=True,
    )
    assert distance.cosine(
        initial_embeddings,
        model.encode(first_completion["choices"][0]["message"]["content"]),
    ) < distance.cosine(
        initial_embeddings,
        model.encode(second_completion["choices"][0]["message"]["content"]),
    )

    assert run_chat_completion(model_name, messages, frequency_penalty=-2.1) == 400
    assert run_chat_completion(model_name, messages, frequency_penalty=2.1) == 400


def test_presence_penalty(model_name):
    messages = [
        {
            "role": "system",
            "content": "You are content maker. Write the response in formal style that appropriately completes the request",
        },
        {
            "role": "user",
            "content": "Write a 1000-word article about the development of computer science",
        },
    ]
    max_tokens = 1500
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    initial_completion = run_chat_completion(
        model_name, messages, return_completion=True
    )
    initial_embeddings = model.encode(
        initial_completion["choices"][0]["message"]["content"]
    )
    messages.append(
        {
            "role": "assistant",
            "content": initial_completion["choices"][0]["message"]["content"],
        }
    )
    messages.append(
        {
            "role": "user",
            "content": "Write a 1000-word article about the development of computer science",
        }
    )
    first_completion = run_chat_completion(
        model_name,
        messages,
        max_tokens=max_tokens,
        frequency_penalty=0.0,
        presence_penalty=-0.0,
        temperature=0.0,
        return_completion=True,
    )
    second_completion = run_chat_completion(
        model_name,
        messages,
        max_tokens=max_tokens,
        frequency_penalty=0.0,
        presence_penalty=2.0,
        temperature=0.0,
        return_completion=True,
    )
    assert distance.cosine(
        initial_embeddings,
        model.encode(first_completion["choices"][0]["message"]["content"]),
    ) < distance.cosine(
        initial_embeddings,
        model.encode(second_completion["choices"][0]["message"]["content"]),
    )

    assert run_chat_completion(model_name, messages, presence_penalty=-2.1) == 400
    assert run_chat_completion(model_name, messages, presence_penalty=2.1) == 400


def test_model_name(model_name):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]
    completion = run_chat_completion(model_name, messages, return_completion=True)
    assert completion["model"] == model_name


def test_choices_exist(model_name):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]
    completion = run_chat_completion(model_name, messages, return_completion=True)
    assert "choices" in completion.keys()
    assert "index" in completion["choices"][0].keys()
    assert "finish_reason" in completion["choices"][0].keys()
    assert "message" in completion["choices"][0].keys()


def test_usage(model_name):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]
    completion = run_chat_completion(model_name, messages, return_completion=True)
    assert "usage" in completion.keys()
    assert "prompt_tokens" in completion["usage"].keys()
    assert "total_tokens" in completion["usage"].keys()
    assert "completion_tokens" in completion["usage"].keys()


def test_id_completion(model_name):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]
    first_completion = run_chat_completion(model_name, messages, return_completion=True)
    second_completion = run_chat_completion(
        model_name, messages, return_completion=True
    )
    assert first_completion["id"] != second_completion["id"]


def test_object_type(model_name):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]
    completion = run_chat_completion(model_name, messages, return_completion=True)
    assert isinstance(completion, OpenAIObject)


def test_created_time(model_name):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]
    # The "created" timestamp is only provided at 1-second
    # granularity, so we shouldn't compare with a finer granularity.
    st_time = int(time.time())
    completion = run_chat_completion(model_name, messages, return_completion=True)
    end_time = int(time.time())
    assert st_time <= completion["created"] <= end_time


@pytest.mark.parametrize(
    "prompt",
    [
        [1, "123"],
        {"text": "How are you?"},
        ("Hello!", "You are a helpful assistant"),
        None,
        10,
        10.5,
        True,
    ],
)
def test_incorrect_content(model_name, prompt):
    messages = [
        {"role": "system", "content": prompt},
    ]

    assert run_chat_completion(model_name, messages) == 400


def test_user_authentication(model_name):
    openai.api_key = "invalid"
    messages = [{"role": "user", "content": "Tell a story about a cat"}]
    assert run_chat_completion(model_name, messages) == 401
    openai.api_key = api_key


def test_cancel_and_follow_up_requests(model_name):
    data = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": "Create a big story about a friendship between a cat and a dog.",
            }
        ],
        "max_tokens": 500,
        "n": 1,
        "stream": False,
        "stop": None,
        "temperature": 0.8,
        "top_p": 1.0,
        "presence_penalty": 0,
        "return_completion": False,
    }
    url = openai.api_base + "/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai.api_key}",
    }
    try:
        requests.post(url, json=data, headers=headers, timeout=1)
    except requests.exceptions.Timeout:
        print("Timeout of request")

    follow_up_request = requests.post(url, json=data, headers=headers).json()
    assert "created" in follow_up_request


def send_request_with_timeout(url, data, headers):
    try:
        requests.post(url, json=data, headers=headers, timeout=1)
    except requests.exceptions.Timeout:
        pass


def test_canceling_requests(model_name):
    data = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": "Create a big story about a friendship between a cat and a dog.",
            }
        ],
        "max_tokens": 1000,
        "n": 1,
        "stream": False,
        "stop": None,
        "temperature": 0.8,
        "top_p": 1.0,
        "presence_penalty": 0,
        "return_completion": False,
    }
    url = openai.api_base + "/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai.api_key}",
    }
    start_time = time.time()
    requests.post(url, json=data, headers=headers)
    first_run_time = time.time() - start_time

    with ThreadPoolExecutor(max_workers=4) as executor:
        for _ in range(64):
            executor.submit(send_request_with_timeout, url, data, headers)

    start_time = time.time()
    requests.post(url, json=data, headers=headers)
    second_run_time = time.time() - start_time

    eps = 5
    assert abs(second_run_time - first_run_time) < eps


@pytest.mark.parametrize("tokens", [30, 50, 70])
def test_completion_tokens(model_name, tokens):
    messages = [
        {"role": "user", "content": f"Explain JavaScript objects in {tokens} tokens or less."}
    ]

    completion = run_chat_completion(
        model_name, messages, temperature=0, top_p=1.0, return_completion=True
    )
    threshold = 10
    assert abs(completion["usage"]["completion_tokens"] - tokens) < threshold


def test_same_completion_len(model_name):
    messages = [
        {"role": "user", "content": "Hello, how can you help me? Answer short."}
    ]
    tokens_set = set()

    for _ in range(4):
        completion = run_chat_completion(
            model_name, messages, temperature=0, top_p=1.0, return_completion=True
        )
        tokens_set.add(completion["usage"]["completion_tokens"])

    assert len(tokens_set) == 1
