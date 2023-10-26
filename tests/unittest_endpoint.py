import requests
import pytest
import os
import re
import testdata.config as cfg


# @pytest.fixture(params=cfg.MODEL_ENDPOINTS)
# def get_model_endpoint(request):
#     return request.model_endpoint

# @pytest.fixture(params=cfg.API_KEY)
# def get_api_key(request, api_key):
#     if api_key is None:
#       raise ValueError("API_KEY not found in the .env file")
#     return request.api_key

# @pytest.fixture(params=cfg.HEADERS)
# def get_headers(request):
#     return request.headers

# @pytest.mark.smoke
# @pytest.mark.parametrize("model_endpoint", cfg.MODEL_ENDPOINTS) 
# @pytest.fixture(params=response, autouse=True)
# def test_availability(response):

#     assert response.status_code == 200
#     yield
#     print("Test passed.")

@pytest.mark.smoke

@pytest.mark.parametrize("headers", cfg.HEADERS)
@pytest.mark.parametrize("model_endpoint", cfg.MODEL_ENDPOINTS)
@pytest.mark.parametrize("prompt, expected_pattern", [
    ("Translate the following text to French: 'Hello, world.'", r"Bonjour"),
    ("Sum the numbers 2 and 3.", r"5"),
])
def test_model_endpoint(model_endpoint, prompt, expected_pattern, headers):
    data = {
        "model": model_endpoint,
        "messages": [
            {
                "role": "user",
                "content": "" # need to fill before use inference
            }
        ],
        "stream": False,
        "max_tokens": 256
    }

    

    data["messages"][0]["content"] = prompt
    response = requests.post(model_endpoint+"/v1/chat/completions", headers=headers, json=data)
    print(response)
    assert response.status_code == 200, f"HTTP Status Code for {model_endpoint}: {response.status_code}"

    assert response.content.strip() != "", f"Response for {model_endpoint} is empty"
    response_data = response.json()
    print(response_data)
    print(response.content)

    
    assert "content" in response_data['choices'][0]["message"]
    
    model_output = response_data['choices'][0]["message"]["content"]
    assert re.search(expected_pattern, model_output)
