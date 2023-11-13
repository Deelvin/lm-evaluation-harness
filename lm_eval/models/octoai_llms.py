import os
import json
import time

import requests

from lm_eval.base import BaseLM

REPEAT_REQUEST_TO_OCTOAI_SERVER = 10


class OctoAIEndpointRunnerBase():
  url_postfix = None

  def __init__(
    self,
    model_name: str="llama-2-70b-chat-int4",
    url: str=None,
    batch_size: int=1,
    top_p: float=1.0,
    temperature: float=0.0,
    prod=True,
    token=None,
  ):
    """
    param model_name: str
        Model name from the list of models supported by OctoAI
    """
    self.model_name = model_name
    self.batch_size=batch_size
    if url is not None:
      self.url = url
    else:
      self.url = self.construct_request_url(prod)

    self.init_msg_header(token)
    self.init_base_msg(top_p, temperature)

  def init_msg_header(self, token):
    # Get the API key from the environment variables

    if token is None: # there is no customized token, try to find in env
      token = os.environ["OCTOAI_TOKEN"]
      if token is None:
        raise ValueError("TOKEN not found.")

    self.headers = {
      "authorization": f"Bearer {token}",
      "content-type": "application/json",
    }

  def init_base_msg(self, top_p, temperature):
    self.base_msg = {
        "model": self.model_name,
        "stream": False,
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

  def get_base_msg(self):
    return self.base_msg

  def call_octoai_inference(self):
    response = requests.post(self.url + self.url_postfix, headers=self.headers, json=self.msg)

    if response.status_code != 200:
      print(f"Error: {response.status_code} - {response.text}")

    return json.loads(response.text)

  def _batcher(self, requests):
    for i in range(0, len(requests), self.batch_size):
      yield requests[i:i + self.batch_size]

  def run(self, requests, results):
    if self.batch_size > 1:
      for batch_idx, request_batch in enumerate(_batcher(requests)):
        try:
          self.model_generate_parallel(request_batch, results)
        except ConnectionError as e:
          print(f"ConnectionError: {e}. Skipping this batch and continuing...")
        print(
          f"\r{(batch_idx + 1) * self.batch_size}/{len(requests)} requests processed",
          end="",
        )
    else:
      for num, request in enumerate(requests):
        try:
          self.model_generate(request, results)
        except ConnectionError as e:
          print(f"ConnectionError: {e}. Skipping this request and continuing...")
        print(f"\r{num}/{len(requests)} requests processed", end="")

  def model_generate(self, request, results):
    success = False
    self.prepare_msg_data(request)
    for _ in range(REPEAT_REQUEST_TO_OCTOAI_SERVER):
      response = self.call_octoai_inference()
      if 'choices' in response.keys():
        success = True
        break
    if success:
      results.append(self.get_result(response))
    else:
      print("ERROR: responce does not have choices. Dummy response was inserted")
      results.append(self.dummy_result())

  def model_generate_batch(self, request_batch, results):
    parallel_results={}
    for id in range(len(request_batch)):
      parallel_results[id]=[]
      self.model_generate(request_batch[id], parallel_results[id])

    # Collect results together
    for id in range(len(request_batch)):
      results.extend(parallel_results[id])

  def model_generate_parallel(self, request_batch, results):
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=self.batch_size) as executor:
      futures = []
      parallel_results = {}
      for id in range(len(request_batch)):
        parallel_results[id]=[]
        futures.append(executor.submit(self.model_generate, request_batch[id], parallel_results[id]))

      for future in concurrent.futures.as_completed(futures):
        try:
          future.result()
        except Exception as exc:
          print(f"Error parallel generating predictions: {exc}")

      # Collect results together
      for id in range(len(request_batch)):
        results.extend(parallel_results[id])

  def get_result(self, response):
    raise NotImplementedError("get_result method is not implemented in base class")

  def dummy_result(self):
    raise NotImplementedError("dummy_result method is not implemented in base class")


class OctoAIEndpointRunnerGreedyUntil(OctoAIEndpointRunnerBase):
  url_postfix = "/v1/chat/completions"

  def __init__(self, **kwargs):
    super().__init__(**kwargs)

    self.msg = self.get_base_msg()

  def prepare_msg_data(self, request):
    inp = request[0]
    # TODO(vvchernov): use until to init stop tokens
    # request_args = request[1]
    # until = request_args["until"]
    self.msg["messages"] = [
        {
            "role": "user",
            "content": inp,
        }
    ]

  def get_result(self, response):
    return response['choices'][0]['message']['content']

  def dummy_result(self):
    return "Dummy response"


class OctoAIEndpointRunnerLogLikelihood(OctoAIEndpointRunnerBase):
  url_postfix = "/v1/completions"

  def __init__(self, **kwargs):
    super().__init__(**kwargs)

    self.msg = self.get_base_msg()

  def prepare_msg_data(self, request):
    self.msg["context"] = request[0]
    self.msg["continuation"] = request[1]

  def get_result(self, response):
    logprob = response["logprob"]
    is_greedy = response["is_greedy"]
    return (logprob, is_greedy)

  def dummy_result(self):
    import sys
    return (-sys.float_info.max, False)


runners = {
  "greedy": OctoAIEndpointRunnerGreedyUntil,
  "loglikelihood": OctoAIEndpointRunnerLogLikelihood,
}

def get_octoai_runner(runner_name: str):
  if not runner_name in runners.keys():
    raise ValueError(f"{runner_name} is not a name of octoai runner")
  return runners[runner_name]


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

    # TODO(vvchernov): control it on high level
    self.time_meas = True

    self.model_name = model_name
    self._batch_size=int(batch_size)
    self.max_batch_size=max_batch_size
    self._device=device

    self.runner_args = {
      "model_name": self.model_name,
      "url": url,
      "batch_size": self._batch_size,
      "top_p": top_p,
      "temperature": temperature,
      "prod": prod,
      "token": token,
    }

  @property
  def eot_token_id(self):
    raise NotImplementedError("No eot token is supported.")

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

  def test(self, runner, requests):
    results = []
    if self.time_meas:
      start_timer = time.time()

    runner.run(requests, results)

    if self.time_meas:
      stop_timer = time.time()
      secs = stop_timer - start_timer
      print(
        f"Full time of predictions measurement: {secs:.2f} sec, {(secs / 60):.2f} min, {(secs / 3600):.2f} hour(s)"
      )

    return results

  def accuracy_test(self, requests, runner_name="greedy"):
    if not requests:
      return []

    runner = get_octoai_runner(runner_name)(**self.runner_args)

    return self.test(runner, requests)

  def loglikelihood(self, requests):
    return self.accuracy_test(requests, "loglikelihood")

  def greedy_until(self, requests):
    return self.accuracy_test(requests)

  def _model_call(self, inps):
    raise NotImplementedError("OctoAI does not support one model call")
