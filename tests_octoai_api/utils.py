import os

import requests
import octoai
from octoai.errors import OctoAIError
from pydantic.v1.error_wrappers import ValidationError


def path_to_file(file_name):
    return os.path.join(os.path.dirname(__file__), file_name)


def run_completion(
    model_name,
    text,
    token,
    endpoint,
    max_tokens=10,
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
        client = octoai.client.Client(
            token=token,
        )
        client.chat.completions.endpoint = endpoint + "/v1/chat/completions"
        completion = client.chat.completions.create(
            model=model_name,
            messages=text,
            max_tokens=max_tokens,
            stream=stream,
            stop=stop,
            top_p=top_p,
            temperature=temperature,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )
        print(completion.dict())
        if return_completion:
            return completion.dict()
    except OctoAIError as e:
        if return_completion:
            raise
        print(e)
        http_response = int(e.message.split()[1])
    except ValidationError as e:
        if return_completion:
            raise
        print(e)
        http_response = 400
    return http_response


def send_request_with_timeout(token, endpoint, data):
    url = endpoint + "/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }
    try:
        requests.post(url, json=data, headers=headers, timeout=1)
    except requests.exceptions.Timeout:
        print("Timeout of request")
    return None


def send_request_get_response(token, endpoint, data):
    client = octoai.client.Client(
        token=token,
    )
    try:
        response = client.infer(
            endpoint_url=endpoint + "/v1/chat/completions",
            inputs=data
        )
        return response
    except OctoAIError as e:
        print(e)
        raise


def model_data(
    model_name,
    message,
    max_tokens=10,
    stream=False,
    stop=None,
    temperature=0.0001,
    top_p=1.0,
    frequency_penalty=0,
    presence_penalty=0,
    n=1,
):
    return {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": message,
            }
        ],
        "max_tokens": max_tokens,
        "stream": stream,
        "stop": stop,
        "temperature": temperature,
        "top_p": top_p,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "n": n,
    }
