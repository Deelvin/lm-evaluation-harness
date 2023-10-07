import requests
import os
import json
import concurrent.futures

import time

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

    self.time_meas = True

    self.model_name = model_name
    self._batch_size=int(batch_size)
    self.max_batch_size=max_batch_size
    self._device=device
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
    return self._batch_size

  @property
  def device(self):
    return self._device

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

    results = []
    if self.time_meas:
      start_timer = time.time()
    if self.batch_size > 1:
      def _batcher(in_requests):
        for i in range(0, len(in_requests), self.batch_size):
          yield in_requests[i:i + self.batch_size]

      for request_batch in _batcher(requests):
        self._model_generate_parallel(request_batch, results)
    else:
      for request in requests:
        inp = request[0]
        request_args = request[1]
        until = request_args["until"]
        self._model_generate(inp, results, stop=until)
    if self.time_meas:
      stop_timer = time.time()
      secs = stop_timer-start_timer
      print("Full time of predictions measurement:", secs, "sec", secs/60, "min", secs/3600, "hour(s)")
    return results

  def call_octoai_inference(self, user_input: str):
    self.data["messages"][1]["content"] = user_input
    response = requests.post(self.url, headers=self.headers, json=self.data)

    if response.status_code != 200:
      print(f"Error: {response.status_code} - {response.text}")

    return response

  def _model_call(self, inps):
    raise NotImplementedError("OctoAI does not support one model call")

  # TODO(vvchernov): do we need additional args? max_tokens, temperature..
  def _model_generate(self, inps, results, stop=[]):
    response = self.call_octoai_inference(inps)
    response = json.loads(response.text)
    results.append(response['choices'][0]['message']['content'])

  def _model_generate_parallel(self, request_batch, results):
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
      futures = []
      for id in range(len(request_batch)):
        inp = request_batch[id][0]
        request_args = request_batch[id][1]
        until = request_args["until"]
        futures.append(executor.submit(self._model_generate, inp, results, stop=until))

      for future in concurrent.futures.as_completed(futures):
        try:
          future.result()
        except Exception as exc:
          raise RuntimeError(f"Error parallel generating predictions: {exc}")
