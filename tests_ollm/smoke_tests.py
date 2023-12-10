import os
import types
import time
from pathlib import Path

import requests
import pytest


import numpy as np
from scipy.spatial import distance
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

import openai
from utils import run_chat_completion, send_request_get_response, send_request_with_timeout, path_to_file

# For compatibility with OpenAI versions before v1.0
# https://github.com/openai/openai-python/pull/677.
OPENAI_VER_MAJ = int(openai.__version__.split(".")[0])

if OPENAI_VER_MAJ >= 1:
    from openai import APIError, AuthenticationError, APIConnectionError
    from pydantic import BaseModel as CompletionObject
else:
    from openai.error import APIError, AuthenticationError, APIConnectionError
    from openai.openai_object import OpenAIObject as CompletionObject

from sentence_transformers import SentenceTransformer

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

# Tests for processing input parameters
# TODO: Add assertion desctiptions
@pytest.mark.input_parameter
def test_valid_model_name(model_name, token, endpoint):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]
    assert run_chat_completion(model_name, messages, token, endpoint) == 200

@pytest.mark.input_parameter
def test_invalid_model_name(model_name, token, endpoint):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]
    model_name += "_dummy_check"
    assert run_chat_completion(model_name, messages, token, endpoint) != 200

@pytest.mark.input_parameter
def test_valid_role(model_name, token, endpoint):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "How are you!"},
    ]

    assert run_chat_completion(model_name, messages, token, endpoint) == 200

@pytest.mark.input_parameter
def test_invalid_role(model_name, token, endpoint):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "How are you!"},
    ]
    
    messages.append({"role": "dummy_role", "content": "dummy_content"})
    assert run_chat_completion(model_name, messages, token, endpoint) != 200

    messages.pop()
    assert run_chat_completion(model_name, messages, token, endpoint) == 200

@pytest.mark.input_parameter
@pytest.mark.parametrize("max_tokens", [10, 100, 300, 500, 1024])
def test_valid_max_tokens(model_name, max_tokens, token, endpoint):
    messages = [
        {
            "role": "system",
            "content": "Below is an instruction that describes a task. Write a response that appropriately completes the request.",
        },
        {"role": "user", "content": "Write a really long blog about Seattle."},
    ]
    completion = run_chat_completion(
        model_name,
        messages,
        token,
        endpoint,
        max_tokens=max_tokens,
        return_completion=True,
    )
    assert 0 < completion["usage"]["completion_tokens"] <= max_tokens
    assert len(completion["choices"][0]["message"]["content"]) > 0

@pytest.mark.input_parameter
@pytest.mark.xfail(
    reason="Due to Internal Server Error (500) hides expected invalid_request_error (400)"
)
def test_invalid_max_tokens(model_name, context_size, token, endpoint):
    messages = [
        {
            "role": "system",
            "content": "Below is an instruction that describes a task. Write a response that appropriately completes the request.",
        },
        {"role": "user", "content": "Write a really long blog about Seattle."},
    ]
    assert (
        run_chat_completion(model_name, messages, token, endpoint, max_tokens=-1) == 400
    )
    assert (
        run_chat_completion(
            model_name, messages, token, endpoint, max_tokens=context_size * 2
        )
        == 400
    )
    completion = run_chat_completion(
        model_name,
        messages,
        token,
        endpoint,
        max_tokens=context_size,
        return_completion=True,
    )
    assert 0 < completion["usage"]["completion_tokens"] <= context_size
    assert len(completion["choices"][0]["message"]["content"]) > 0

@pytest.mark.input_parameter
@pytest.mark.xfail(reason="Need to validate distance measurement approach")
def test_valid_temperature(model_name, token, endpoint):
    """The higher the temperature, the further the distance from the expected."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a blog about Seattle"},
    ]
    max_tokens = 784
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    goldev_embed = np.load(path_to_file("golden_temp_0.npy"))

    distances = []
    for temperature in [0.0, 1.0, 2.0]:
        completion = run_chat_completion(
            model_name,
            messages,
            token,
            endpoint,
            max_tokens=max_tokens,
            temperature=temperature,
            return_completion=True,
        )
        curr_embeddings = model.encode(completion["choices"][0]["message"]["content"])
        distances.append(distance.cosine(curr_embeddings, goldev_embed))

    assert distances == sorted(distances)

@pytest.mark.input_parameter
@pytest.mark.parametrize("temperature", [-0.1, 2.1])
def test_invalid_temperature(model_name, temperature, token, endpoint):
    """Invalid temperatures should produce an error.

    Temperature is allowed to range from 0 to 2.0. Outside of this
    range, an error should be returned in the completion.
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a blog about Seattle"},
    ]

    with pytest.raises(APIError):
        completion = run_chat_completion(
            model_name,
            messages,
            token,
            endpoint,
            temperature=temperature,
            return_completion=True,
        )

@pytest.mark.input_parameter
@pytest.mark.xfail(reason="Need to validate distance measurement approach")
def test_valid_top_p(model_name, token, endpoint):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a blog about Seattle"},
    ]
    max_tokens = 784
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    goldev_embed = np.load(path_to_file("golden_top_p_0.npy"))
    completion = run_chat_completion(
        model_name,
        messages,
        token,
        endpoint,
        max_tokens=max_tokens,
        top_p=0.00001,
        return_completion=True,
    )
    curr_embeddings = model.encode(completion["choices"][0]["message"]["content"])
    prev_dist = distance.cosine(curr_embeddings, goldev_embed)

    for top_p in [0.2, 1.0]:
        completion = run_chat_completion(
            model_name,
            messages,
            token,
            endpoint,
            max_tokens=max_tokens,
            top_p=top_p,
            return_completion=True,
        )
        curr_embeddings = model.encode(completion["choices"][0]["message"]["content"])
        cur_distance = distance.cosine(curr_embeddings, goldev_embed)
        assert prev_dist <= cur_distance
        prev_dist = cur_distance

@pytest.mark.input_parameter
@pytest.mark.parametrize("top_p", [-0.1, 1.1])
def test_invalid_top_p(model_name, top_p, token, endpoint):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a blog about Seattle"},
    ]

    assert (
        run_chat_completion(model_name, messages, token, endpoint, top_p=top_p) == 400
    )

@pytest.mark.input_parameter
@pytest.mark.parametrize("n", [1, 5, 10])
def test_valid_number_chat_completions(model_name, n, token, endpoint):
    if n > 1:
        pytest.xfail("Multiple outputs is not supported yet")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]
    completion = run_chat_completion(
        model_name, messages, token, endpoint, n=n, return_completion=True
    )
    assert len(completion["choices"]) == n

def test_invalid_number_chat_completions(model_name, token, endpoint):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]
    assert run_chat_completion(
        model_name, messages, token, endpoint, n=0
    ) == 400

@pytest.mark.input_parameter
@pytest.mark.xfail(reason="When stream = True last chunks consist of empty strings")
def test_stream(model_name, token, endpoint):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a blog about Seattle"},
    ]

    start_time = time.time()
    completion_stream = run_chat_completion(
        model_name,
        messages,
        token,
        endpoint,
        stream=True,
        temperature=0.0,
        return_completion=True,
    )
    time_to_first_token = time.time() - start_time
    assert isinstance(completion_stream, types.GeneratorType)
    stream_str = ""
    for chunk in completion_stream:
        chunk_data = chunk["choices"][0]["delta"]["content"]
        if (
            chunk["choices"][0]["delta"]["role"] == "assistant"
            and chunk_data is not None
        ):
            if chunk["choices"][0]["finish_reason"] != "stop":
                assert chunk_data != '', "Recieved chunk consists of empty string"
            stream_str += chunk_data
    start_time = time.time()
    completion = run_chat_completion(
        model_name,
        messages,
        token,
        endpoint,
        stream=False,
        temperature=0.0,
        return_completion=True,
    )
    full_response_time = time.time() - start_time

    assert stream_str == completion["choices"][0]["message"]["content"]
    assert full_response_time > time_to_first_token

@pytest.mark.input_parameter
@pytest.mark.xfail(
    reason="Need upstream with OpenAI approach (return completion without stop token)"
)
@pytest.mark.parametrize("stop", [["tomato", "tomatoes"], [".", "!"]])
def test_valid_stop(model_name, stop, token, endpoint):
    messages = [{"role": "user", "content": "How to cook tomato paste?"}]
    completion = run_chat_completion(
        model_name,
        messages,
        token,
        endpoint,
        max_tokens=300,
        stop=stop,
        return_completion=True,
    )
    for seq in stop:
        assert (
            seq not in completion["choices"][0]["message"]["content"]
        ) and completion["choices"][0]["finish_reason"] == "stop"

    # These test assumes that stop token return with completion
    # assert completion["choices"][0]["finish_reason"] == "stop"
    # text = completion["choices"][0]["message"]["content"]
    # if "." in stop:
    #     assert text[-1] in stop
    #     for seq in stop:
    #         assert seq not in text[:-1]
    # else:
    #     words = text.split()
    #     assert words[-1] in stop
    #     for seq in stop:
    #         assert seq not in words[:-1]

@pytest.mark.input_parameter
@pytest.mark.parametrize("stop", [42, {"stop": "word"}, [1, 2, 3]])
def test_invalid_stop(model_name, stop, token, endpoint):
    messages = [{"role": "user", "content": "How to cook tomato paste?"}]
    assert run_chat_completion(
        model_name,
        messages,
        token,
        endpoint,
        stop=stop,
    ) == 400

@pytest.mark.input_parameter
@pytest.mark.parametrize("prompt", ["Hi!", None])
def test_valid_content(model_name, prompt, token, endpoint):
    message = {"role": "system", "content": prompt}

    assert run_chat_completion(model_name, [message], token, endpoint) == 200

@pytest.mark.input_parameter
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
def test_invalid_content(model_name, prompt, token, endpoint):
    message = {"role": "system", "content": prompt}

    assert run_chat_completion(model_name, [message], token, endpoint) in [422, 400]

@pytest.mark.input_parameter
@pytest.mark.xfail(reason="Frequency penalty has not been implemented yet")
def test_valid_frequency_penalty(model_name, token, endpoint):
    messages = [
        {
            "role": "system",
            "content": "You are content maker. Write the response in formal style that appropriately completes the request",
        },
        {
            "role": "user",
            "content": "Write a 800-word article about large language models using the word 'transformer' as often as possible",
        },
    ]
    responses = []
    for frequency_penalty in [-2, -1, 0, 1, 2]:
        responses.append(
            run_chat_completion(
                model_name, 
                messages, 
                token, 
                endpoint, 
                frequency_penalty=frequency_penalty,
                max_tokens=800,
                return_completion=True
        )["choices"][0]["message"]["content"]
    )
    for i in range(1, 4):
        assert responses[i].lower().count("transformer") < responses[i - 1].lower().count("transformer")

@pytest.mark.input_parameter
@pytest.mark.parametrize("fr_pen", [-2.1, 2.1])
def test_invalid_frequency_penalty(model_name, fr_pen, token, endpoint):
    messages = [
        {
            "role": "system",
            "content": "You are content maker. Write the response in formal style that appropriately completes the request",
        },
        {
            "role": "user",
            "content": "Write a 800-word article about large language models using the word 'transformer' as often as possible",
        },
    ]
    initial_completion = run_chat_completion(
        model_name, messages, token, endpoint, return_completion=True
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
            "content": "Write a 800-word article about large language models using the word 'transformer' as often as possible",
        }
    )

    assert run_chat_completion(
        model_name, messages, token, endpoint, frequency_penalty=fr_pen
    ) == 400

@pytest.mark.input_parameter
@pytest.mark.xfail(reason="Presence penalty has not been implemented yet")
def test_valid_presence_penalty(model_name, token, endpoint):
    messages = [
        {
            "role": "system",
            "content": "You are content maker. Write the response in formal style that appropriately completes the request",
        },
        {
            "role": "user",
            "content": "Write a 800-word article about large language models using the word 'transformer' as often as possible",
        },
    ]
    responses = []
    for presence_penalty in [-2, -1, 0, 1, 2]:
        responses.append(
            run_chat_completion(
                model_name, 
                messages, 
                token, 
                endpoint, 
                presence_penalty=presence_penalty,
                max_tokens=800,
                return_completion=True
        )["choices"][0]["message"]["content"]
    )
    for i in range(1, 4):
        assert responses[i].lower().count("transformer") < responses[i - 1].lower().count("transformer")

@pytest.mark.input_parameter
@pytest.mark.parametrize("pr_pen", [-2.1, 2.1])
def test_invalid_frequency_penalty(model_name, pr_pen, token, endpoint):
    messages = [
        {
            "role": "system",
            "content": "You are content maker. Write the response in formal style that appropriately completes the request",
        },
        {
            "role": "user",
            "content": "Write a 800-word article about large language models using the word 'transformer' as often as possible",
        },
    ]
    initial_completion = run_chat_completion(
        model_name, messages, token, endpoint, return_completion=True
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
            "content": "Write a 800-word article about large language models using the word 'transformer' as often as possible",
        }
    )

    assert run_chat_completion(
        model_name, messages, token, endpoint, presence_penalty=pr_pen
    ) == 400

# Tests for correctness of response
@pytest.mark.response_correctness
def test_response_model_name(model_name, token, endpoint):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]
    completion = run_chat_completion(
        model_name, messages, token, endpoint, return_completion=True
    )
    assert completion["model"] == model_name

@pytest.mark.response_correctness
def test_response_choices(model_name, token, endpoint):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]
    completion = run_chat_completion(
        model_name, messages, token, endpoint, return_completion=True
    )
    assert "choices" in completion.keys()
    assert "index" in completion["choices"][0].keys()
    assert "finish_reason" in completion["choices"][0].keys()
    assert "message" in completion["choices"][0].keys()

@pytest.mark.response_correctness
def test_response_usage(model_name, token, endpoint):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]
    completion = run_chat_completion(
        model_name, messages, token, endpoint, return_completion=True
    )
    assert "usage" in completion.keys()
    assert "prompt_tokens" in completion["usage"].keys()
    assert "total_tokens" in completion["usage"].keys()
    assert "completion_tokens" in completion["usage"].keys()

@pytest.mark.response_correctness
def test_response_id(model_name, token, endpoint):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]
    first_completion = run_chat_completion(
        model_name, messages, token, endpoint, return_completion=True
    )
    second_completion = run_chat_completion(
        model_name, messages, token, endpoint, return_completion=True
    )
    assert first_completion["id"] != second_completion["id"]

@pytest.mark.response_correctness
def test_response_object_type(model_name, token, endpoint):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]
    completion = run_chat_completion(
        model_name, messages, token, endpoint, return_completion=True
    )
    assert isinstance(completion, CompletionObject)

@pytest.mark.response_correctness
def test_response_created_time(model_name, token, endpoint):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]
    # The "created" timestamp is only provided at 1-second
    # granularity, so we shouldn't compare with a finer granularity.
    st_time = int(time.time())
    completion = run_chat_completion(
        model_name, messages, token, endpoint, return_completion=True
    )
    end_time = int(time.time())
    assert st_time <= completion["created"] <= end_time

# Test for authentification
@pytest.mark.auth
def test_invalid_token_authentification(model_name, endpoint):
    messages = [{"role": "user", "content": "Tell a story about a cat"}]
    assert run_chat_completion(model_name, messages, "invalid", endpoint) == 401
