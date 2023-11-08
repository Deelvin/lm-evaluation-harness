import os
import json
import time

import requests

from lm_eval.base import BaseLM

REPEAT_REQUEST_TO_OCTOAI_SERVER = 10

class OctoAIEndpointLM(BaseLM):
  def __init__(
    self,
    model_name: str="llama-2-70b-chat-int4",
    url: str=None,
    batch_size: int=1,
    max_batch_size: int=None,
    device: str=None,
    top_p: float=1.0,
    temperature: float=0.0,
    prod=True,
    token=None,
  ):
    """
    :param model_name: str
        Model name from the list of models supported by OctoAI
    """
    super().__init__()

    self.time_meas = True

    self.model_name = model_name
    self._batch_size = int(batch_size)
    self.max_batch_size = max_batch_size
    self._device = device

    self.init_remote(url, top_p, temperature, prod, token)

  def init_remote(self, url, top_p, temperature, prod, token):
    # Get the API key from the environment variables

    if token is None: # there is no customized token, try to find in env
      token = os.environ["OCTOAI_TOKEN"]
      if token is None:
        raise ValueError("TOKEN not found.")

    if url is not None:
      self.url = url
    elif prod:
      self.url = "https://text.octoai.run"
    else:
      self.url = "https://text.customer-endpoints.nimbus.octoml.ai"

    self.headers = {
      "authorization": f"Bearer {token}",
      "content-type": "application/json",
    }

    self.data = {
      "model": self.model_name,
      "messages": [{"role": "user", "content": ""}],  # need to fill before use inference
      # "stream": False,
      "max_tokens": 256,
      "top_p": top_p,
      "temperature": temperature,
    }

  def construct_request_url(self, prod):
    if prod:
      url = "https://text.octoai.run"
    else:
      url = "https://text.customer-endpoints.nimbus.octoml.ai"

    return url

  @property
  def eot_token_id(self):
    raise NotImplementedError

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
    raise NotImplementedError("Cannot call tokenizer by API")

  def tok_decode(self, tokens):
    raise NotImplementedError("Cannot call tokenizer by API")

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
          yield in_requests[i : i + self.batch_size]

      for batch_idx, request_batch in enumerate(_batcher(requests)):
        try:
          self._model_generate_parallel(request_batch, results)
        except ConnectionError as e:
          print(f"ConnectionError: {e}. Skipping this batch and continuing...")
        print(
          f"\r{(batch_idx + 1) * self.batch_size}/{len(requests)} requests processed",
          end="",
        )
    else:
      for num, request in enumerate(requests):
        inp = request[0]
        request_args = request[1]
        until = request_args["until"]
        try:
          self._model_generate(inp, results, stop=until)
        except ConnectionError as e:
          print(f"ConnectionError: {e}. Skipping this request and continuing...")
        print(f"\r{num}/{len(requests)} requests processed", end="")

    if self.time_meas:
      stop_timer = time.time()
      secs = stop_timer - start_timer
      print(
        f"Full time of predictions measurement: {secs:.2f} sec, {(secs / 60):.2f} min, {(secs / 3600):.2f} hour(s)"
      )

    return results

  def call_octoai_reset(self):
    try:
      resp = requests.post(self.url + "/chat/reset", headers=self.headers)
      return resp.json()
    except Exception as e:
      print(f"Error resetting chat for endpoint {self.url}")
      print(e)
      return

  def call_octoai_inference(self, user_input: str):
    self.data["messages"][0]["content"] = user_input
    response = requests.post(
      self.url + "/v1/chat/completions", headers=self.headers, json=self.data
    )

    if response.status_code != 200:
      print(f"Error: {response.status_code} - {response.text}")

    return response

  def _model_call(self, inps):
    raise NotImplementedError("OctoAI does not support one model call")

  # TODO(vvchernov): do we need additional args? max_tokens, temperature..
  def _model_generate(self, inps, results, stop=[]):
    success = False
    for _ in range(REPEAT_REQUEST_TO_OCTOAI_SERVER):
      # TODO(vvchernov): process wrong reset
      self.call_octoai_reset()
      response = self.call_octoai_inference(inps)
      response = json.loads(response.text)
      if "choices" in response.keys():
        success = True
        break
    if success:
      results.append(response["choices"][0]["message"]["content"])
    else:
      print("ERROR: responce does not have choices. Dummy response was inserted")
      results.append("Dummy response")

  def _model_generate_batch(self, request_batch, results):
    parallel_results = {}
    for requiest_id in range(len(request_batch)):
      parallel_results[requiest_id] = []
      inp = request_batch[requiest_id][0]
      request_args = request_batch[requiest_id][1]
      until = request_args["until"]
      self._model_generate(inp, parallel_results[requiest_id], stop=until)

    # Collect results together
    for requiest_id in range(len(request_batch)):
      results.extend(parallel_results[requiest_id])

  def _model_generate_parallel(self, request_batch, results):
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=self.batch_size) as executor:
      futures = []
      parallel_results = {}
      for id in range(len(request_batch)):
        parallel_results[id] = []
        inp = request_batch[id][0]
        request_args = request_batch[id][1]
        until = request_args["until"]
        futures.append(
            executor.submit(self._model_generate, inp, parallel_results[id], stop=until)
        )

      for future in concurrent.futures.as_completed(futures):
        try:
          future.result()
        except Exception as exc:
          print(f"Error parallel generating predictions: {exc}")

      # Collect results together
      for id in range(len(request_batch)):
        results.extend(parallel_results[id])
