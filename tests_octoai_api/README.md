## Install dependencies
First of all, install octoai-sdk

`python3 -m pip install octoai-sdk`

## The difference between the octoai api and the openai api

Currently, tests with the number of chat completions are skipped because there is no support for this parameter in `client.chat.completions.create`.

The list of these tests:

**smoke_tests**
- `test_invalid_temperature`
- `test_valid_number_chat_completions`
- `test_invalid_number_chat_completions`

**extended_tests**
- `test_large_number_chat_completions`

**Other differences:**

In the openai api, `test_invalid_stop` consider that the `int(=42)` and `dict(={"stop": "tomato"})` can't be a stop word, but in the octoai api it can be.

In the openai api, `test_valid_content` consider that the `None` can be a message, but in the octoai api it can't be.