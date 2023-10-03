import requests
import os
import json

from lm_eval.base import BaseLM

# Start line
# python3 main.py --model=octoai --tasks=math_algebra --batch_size=1 --output_path=./results_alg.json --device cuda:0 --limit 0.1
# need --model_args="" with model name while hardcode

class OctoAIEndpointLM(BaseLM):
  def __init__(
      self,
      model_name="llama-2-70b-chat",
      batch_size=1,
      max_batch_size=None,
      device=None):
    """
    :param model_name: str
        Model name from the list of models supported by OctoAI
    """
    super().__init__()

    self.model_name = model_name
    self.batch_size=int(batch_size)
    self.max_batch_size=max_batch_size
    self.device=device
    # TODO(vvchernov): check that model name is supported

    self.init_remote()

  def init_remote(self):
    # TODO(vvchernov): possibly not-safe approach need to get key each time
    # Get the API key from the environment variables
    api_key=os.environ["OCTOAI_API_KEY"]

    if api_key is None:
      raise ValueError("API_KEY not found in the .env file")

    # TODO(vvchernov): looks like hard code
    self.url = "https://llama-2-70b-chat-demo-kk0powt97tmb.octoai.run/v1/chat/completions"

    self.headers = {
      "accept": "text/event-stream",
      "authorization": f"Bearer {api_key}",
      "content-type": "application/json",
    }

    self.data = {
        "model": self.model_name,
        "messages": [
            {
                "role": "assistant",
                "content": "Below is an instruction that describes a task. Write a response that appropriately completes the request."
            },
            {
                "role": "user",
                "content": "" # need to fill before use inference
            }
        ],
        "stream": False,
        "max_tokens": 256
    }

  @property
  def eot_token_id(self):
    raise NotImplementedError("No idea about anthropic tokenization.")

  @property
  def max_length(self):
    return 2048

  @property
  def max_gen_toks(self):
    return 256

  @property
  def batch_size(self):
      # Isn't used because we override _loglikelihood_tokens
      raise NotImplementedError()

  @property
  def device(self):
      # Isn't used because we override _loglikelihood_tokens
      raise NotImplementedError()

  def tok_encode(self, string: str):
      return string
      #raise NotImplementedError("No idea about anthropic tokenization.")

  def tok_decode(self, tokens):
      return tokens
      #raise NotImplementedError("No idea about anthropic tokenization.")

  def _loglikelihood_tokens(self, requests, disable_tqdm=False):
    raise NotImplementedError("No support for logits.")

  def greedy_until(self, requests):
    if not requests:
      return []

    res = []
    for request in requests:
      inp = request[0]
      request_args = request[1]
      until = request_args["until"]
      # TODO(vvchernov): do we need additional args? max_tokens, temperature..
      response = self._model_generate(inp, stop=until)
      res.append(response)
    return res

  def call_octoai_inference(self, user_input: str):
    self.data["messages"][1]["content"] = user_input
    response = requests.post(self.url, headers=self.headers, json=self.data)

    if response.status_code != 200:
      print(f"Error: {response.status_code} - {response.text}")

    return response

  def _model_call(self, inps):
    raise NotImplementedError("OctoAI does not support one model call")

  def _model_generate(self, inps, stop):
    response = self.call_octoai_inference(inps)
    response = json.loads(response.text)
    return response['choices'][0]['message']['content']
