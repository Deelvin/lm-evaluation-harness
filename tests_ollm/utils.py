import os

import requests
import openai

# For compatibility with OpenAI versions before v1.0
# https://github.com/openai/openai-python/pull/677.
OPENAI_VER_MAJ = int(openai.__version__.split('.', maxsplit=1)[0])

if OPENAI_VER_MAJ >= 1:
    from openai import APIError, AuthenticationError, APIConnectionError
else:
    from openai.error import APIError, AuthenticationError, APIConnectionError


def path_to_file(file_name):
    return os.path.join(os.path.dirname(__file__), file_name)

def run_completion(
    model_name,
    text,
    token,
    endpoint,
    chat=True,
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
        openai.base_url = endpoint + "/v1"
        if chat is True:
            if OPENAI_VER_MAJ >= 1:
                client = openai.OpenAI(
                    api_key=token,
                )
                chat_completions = client.chat.completions
            else:
                openai.api_base = endpoint + "/v1"
                chat_completions = openai.ChatCompletion
            completion = chat_completions.create(
                model=model_name,
                messages=text,
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
            if OPENAI_VER_MAJ >= 1:
                if chat is False:
                    raise NotImplementedError(
                        "Completion is not supported on new OpenAI API yet"
                    )
            completion = openai.Completion.create(
                model=model_name,
                prompt=text,
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
        print("Timeout of request")
    return None


def send_request_get_response(url, data, headers):
    response = requests.post(url, json=data, headers=headers, timeout=None)
    return response
