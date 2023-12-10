import os
import types
import time
from pathlib import Path

import requests


import numpy as np
from scipy.spatial import distance
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

import openai

# For compatibility with OpenAI versions before v1.0
# https://github.com/openai/openai-python/pull/677.
OPENAI_VER_MAJ = int(openai.__version__.split(".")[0])

if OPENAI_VER_MAJ >= 1:
    from openai import APIError, AuthenticationError, APIConnectionError
    from pydantic import BaseModel as CompletionObject
else:
    from openai.error import APIError, AuthenticationError, APIConnectionError
    from openai.openai_object import OpenAIObject as CompletionObject


def path_to_file(file_name):
    return os.path.join(os.path.dirname(__file__), file_name)

def run_chat_completion(
    model_name,
    messages,
    token,
    endpoint,
    max_tokens=10,
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
    openai.api_key = token
    try:
        if OPENAI_VER_MAJ > 0:
            openai.base_url = endpoint + "/v1"
            client = openai.OpenAI(
                api_key=token,
            )
            completion = client.chat.completions.create(
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
        else:
            openai.api_base = endpoint + "/v1"
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
            if OPENAI_VER_MAJ >= 1:
                return completion.model_dump(exclude_unset=True)
            else:
                return completion
    except (APIError, AuthenticationError, APIConnectionError) as e:
        if return_completion:
            raise
        if OPENAI_VER_MAJ > 0:
            print(e.message)
            http_response = e.status_code
        else:
            print(e.user_message)
            http_response = e.http_status

    return http_response

def send_request_with_timeout(url, data, headers):
    try:
        requests.post(url, json=data, headers=headers, timeout=1)
    except requests.exceptions.Timeout:
        return None


def send_request_get_response(url, data, headers):
    response = requests.post(url, json=data, headers=headers)
    return response
