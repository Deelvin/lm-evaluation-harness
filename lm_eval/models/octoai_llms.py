import os
import json
import time
import json

import requests
import asyncio
from lm_eval.base import BaseLM
import aiohttp
REPEAT_REQUEST_TO_OCTOAI_SERVER = 10


class OctoAIEndpointRunnerBase():
  def __init__(
    self,
    model_name: str="llama-2-70b-chat-int4",
    url: str=None,
    url_postfix: str = "/v1/chat/completions",
    batch_size: int=1,
    max_tokens: int=2048,
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
    self.url_postfix = url_postfix

    self.init_msg_header(token)
    self.init_base_msg(max_tokens, top_p, temperature)

  def init_msg_header(self, token):
    # Get the API key from the environment variables

    if token is None: # there is no customized token, try to find in env
      token = os.environ.get("OCTOAI_TOKEN", "dummy_token")

    self.headers = {
      "authorization": f"Bearer {token}",
      "content-type": "application/json",
    }

  def init_base_msg(self, max_tokens, top_p, temperature):
    self.base_msg = {
        "model": self.model_name,
        "stream": False,
        "max_tokens": max_tokens,
        "stop": ["<|eot_id|>"] if os.environ.get("CONVERSATION_TEMPLATE", "default") == "llama3" else [],
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
      for batch_idx, request_batch in enumerate(self._batcher(requests)):
        try:
          asyncio.run(self.model_generate_parallel(request_batch, results))
        except json.decoder.JSONDecodeError as exc:
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
      if self.response_check(response):
        success = True
        break
    if success:
      results.append(self.get_result(response))
    else:
      print("ERROR: response check failed. Dummy response was inserted")
      results.append(self.dummy_result())

  def model_generate_batch(self, request_batch, results):
    parallel_results={}
    for id in range(len(request_batch)):
      parallel_results[id]=[]
      self.model_generate(request_batch[id], parallel_results[id])

    # Collect results together
    for id in range(len(request_batch)):
      results.extend(parallel_results[id])

  async def model_generate_async(self, request, results):
    success = False
    self.prepare_msg_data(request)
    async with aiohttp.ClientSession() as session:
      exception = None
      for _ in range(REPEAT_REQUEST_TO_OCTOAI_SERVER):
        async with session.post(self.url+ self.url_postfix, headers=self.headers, json=self.msg) as response:
          try:
            response_text = await response.text()
            response_text = json.loads(response_text)
          except Exception as exc:
            exception = str(exc)
            response_text = ""
            continue
          if self.response_check(response_text):
            success = True
            break
      if success:
        results.append(self.get_result(response_text))
      else:
        print("ERROR: response check failed. Dummy response was inserted")
        if exception is not None:
          results.append(exception)
        else:
          results.append(self.dummy_result())

  async def model_generate_parallel(self, request_batch, results):
    parallel_results = {}
    for id in range(len(request_batch)):
      parallel_results[id]=[]
    tasks = [self.model_generate_async(request_batch[id], parallel_results[id]) for id in range(len(request_batch))]
    await asyncio.gather(*tasks)
    for id in range(len(request_batch)):
      results.extend(parallel_results[id])

  def prepare_msg_data(self, request):
    raise NotImplementedError("prepare_msg_data method is not implemented in base class")

  def response_check(self, response):
    return "choices" in response.keys()

  def get_result(self, response):
    raise NotImplementedError("get_result method is not implemented in base class")

  def dummy_result(self):
    raise NotImplementedError("dummy_result method is not implemented in base class")


class OctoAIEndpointRunnerGreedyUntil(OctoAIEndpointRunnerBase):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

    self.msg = self.get_base_msg()

  def prepare_msg_data(self, request):
    inp = request[0]
    # TODO(vvchernov): use until to init stop tokens
    # request_args = request[1]
    # until = request_args["until"]
    if self.url_postfix == "/v1/chat/completions":
      self.msg["messages"] = [
          {
              "role": "user",
              "content": inp,
          }
      ]
    else:  # "v1/completion"
      self.msg["prompt"] = inp

  def get_result(self, response):
    return response["choices"][0]["message"]["content"]

  def dummy_result(self):
    return "Dummy response"


class OctoAIEndpointRunnerLogLikelihood(OctoAIEndpointRunnerBase):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

    self.msg = self.get_base_msg()

  def prepare_msg_data(self, request):
    self.context = request[0]
    self.continuation = request[1]
    if self.url_postfix == "/v1/chat/completions":
      self.msg["messages"] = [
          {
              "role": "user",
              "content": self.context + self.continuation,
          }
      ]
    else:  # "v1/completion"
      self.msg["prompt"] = self.context + self.continuation
    self.msg["loglikelihood"] = True

  def model_generate(self, request, results):
    success = False
    self.prepare_msg_data(request)
    for _ in range(REPEAT_REQUEST_TO_OCTOAI_SERVER):
      response = self.call_octoai_inference()
      if self.response_check(response):
        success = True
        break
    if success:
      results.append(self.get_result(response, request[0], request[1]))
    else:
      print("ERROR: response check failed. Dummy response was inserted")
      results.append(self.dummy_result())

  async def model_generate_async(self, request, results):
    success = False
    self.prepare_msg_data(request)
    async with aiohttp.ClientSession() as session:
      exception = None
      for _ in range(REPEAT_REQUEST_TO_OCTOAI_SERVER):
        async with session.post(self.url+ self.url_postfix, headers=self.headers, json=self.msg) as response:
          try:
            response_text = await response.text()
            response_text = json.loads(response_text)
          except Exception as exc:
            exception = str(exc)
            response_text = ""
            continue
          if self.response_check(response_text):
            success = True
            break
      if success:
        results.append(self.get_result(response_text, request[0], request[1]))
      else:
        print("ERROR: response check failed. Dummy response was inserted")
        if exception is not None:
          results.append(exception)
        else:
          results.append(self.dummy_result())

  def get_llama_token(self, token: str):
    res = token
    # Special symbol from tokenizer like underbar (Llama2-style)
    sym = bytes.fromhex("e29681").decode("utf-8")
    # workaround for case sym + "_"
    if token.startswith("_" + sym):
      res = token.replace("_" + sym, "  ", 1)
    elif token.startswith(sym):
      res = token.replace(sym, " ")
    return res

  def get_result(self, response, context, continuation):
    logprob_content = response["choices"][0]["logprobs"]["content"]
    logprobs = []
    tokens = []
    top1_tokens = []
    for content in logprob_content:
      tokens.append(content["token"])
      logprobs.append(content["logprob"])
      top1_tokens.append(content["top_logprobs"][0]["token"])

    # Calculate continuation length
    cont_len = 1
    prob_ctx = context + continuation
    # TODO(vvchernov): support all model types
    token = self.get_llama_token(tokens[-cont_len])
    prob_cont = ""
    while prob_ctx.endswith(token):
      prob_cont = token + prob_cont
      if continuation == prob_cont:
        break
      prob_ctx = prob_ctx[:-len(token)]
      cont_len += 1
      token = self.get_llama_token(tokens[-cont_len])
    try:
      assert continuation.startswith(token), f"Tokenization issue, wrong token: \"{token}\""
    except:
      print("CONTEXT:", context)
      print("CONTINUATION:", continuation)
      print("TOKENS:", tokens)
      print("TOKEN:", f"\"{token}\"")
      return self.dummy_result()

    res_logprob = sum(logprobs[-cont_len:])
    tokens_len = len(tokens)
    res_is_greedy = True
    for i in range(tokens_len - cont_len, tokens_len):
      if top1_tokens[i] != tokens[i]:
        res_is_greedy = False
        break
    return (res_logprob, res_is_greedy)

  def dummy_result(self):
    import sys
    return (-sys.float_info.max, False)


runners_available = {
  "greedy": OctoAIEndpointRunnerGreedyUntil,
  "loglikelihood": OctoAIEndpointRunnerLogLikelihood,
}

def get_octoai_runner(runner_name: str):
  if not runner_name in runners_available.keys():
    raise ValueError(f"{runner_name} is not a name of available octoai runner")
  return runners_available[runner_name]


class OctoAIEndpointLM(BaseLM):
  def __init__(
    self,
    model_name: str="llama-2-70b-chat-int4",
    url: str=None,
    url_postfix: str = "/v1/chat/completions",
    batch_size: int=1,
    max_batch_size: int=None,
    device: str=None,
    max_tokens: int=1024,
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
      "url_postfix": url_postfix,
      "batch_size": self._batch_size,
      "max_tokens": max_tokens,
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
    raise NotImplementedError("OctoAI does not support one model call, loglikelyhood method was override")

  def _model_generate(self, context, max_length, eos_token_id):
    raise NotImplementedError("OctoAI does not support model generate, greedy_until method was override")
