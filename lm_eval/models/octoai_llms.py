import requests
import os
import json
import time

from lm_eval.base import BaseLM

REPEAT_REQUEST_TO_OCTOAI_SERVER = 10

model_urls = {
  # TODO(vvchernov): it is demo, may be need to remove
  "llama-2-7b-chat-hf-s2q0f16":	"https://text-demo-mlc-serve-llama-2-7b-chc591cb-l6z2ijkgynn7.octoai.run/llama-2-7b-chat-hf-s2q0f16",
  "codellama-13b-instruct-fp16": "https://text-mlc-serve-codellama-13b-inst57fdc4-l6z2ijkgynn7.octoai.run/codellama-13b-instruct-fp16",
  "codellama-34b-instruct-int4": "https://text-mlc-serve-codellama-34b-inst087581-l6z2ijkgynn7.octoai.run/codellama-34b-instruct-int4",
  "codellama-34b-instruct-int4-1": "https://text-test-codellama-34b-instruct-int4-1-ht5iv0iul7xi.octoai.run/codellama-34b-instruct-int4",
  "codellama-34b-instruct-fp16": "https://text-mlc-serve-codellama-34b-inst73411d-l6z2ijkgynn7.octoai.run/codellama-34b-instruct-fp16",
  "codellama-7b-instruct-fp16":	"https://text-mlc-serve-codellama-7b-instr48ac9b-l6z2ijkgynn7.octoai.run/codellama-7b-instruct-fp16",
  "llama-2-13b-chat-fp16": "https://text-mlc-serve-llama-2-13b-chat-fp16-l6z2ijkgynn7.octoai.run/llama-2-13b-chat-fp16",
  "llama-2-70b-chat-fp16": "https://text-mlc-serve-llama-2-70b-chat-fp16-l6z2ijkgynn7.octoai.run/llama-2-70b-chat-fp16",
  "llama-2-70b-chat-int4": "https://text-mlc-serve-llama-2-70b-chat-int4-l6z2ijkgynn7.octoai.run/llama-2-70b-chat-int4",
  "llama-2-70b-chat-int4-1": "https://text-test-llama-2-70b-chat-int4-1-ht5iv0iul7xi.octoai.run/llama-2-70b-chat-int4",
  "mistral-7b-instruct-v0.1-fp16": "https://text-mlc-serve-mistral-7b-instruct-fp16-l6z2ijkgynn7.octoai.run/mistral-7b-instruct-v0.1-fp16",
}


class OctoAIEndpointLM(BaseLM):
  def __init__(
      self,
      model_name="llama-2-70b-chat-int4",
      batch_size=1,
      max_batch_size=None,
      device=None,
      top_p=1,
      temperature=0.0,
      prod=True,
  ):
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

    self.init_remote(top_p, temperature, prod)

  def init_remote(self, top_p, temperature, prod):
    # Get the API key from the environment variables
    api_key=os.environ["OCTOAI_API_KEY"]

    if api_key is None:
      raise ValueError("API_KEY not found in the .env file")

    self.url = self.construct_request_url(prod)

    self.headers = {
      # "accept": "text/event-stream",
      "authorization": f"Bearer {api_key}",
      "content-type": "application/json",
    }

    self.data = {
        "model": self.model_name,
        "messages": [
            {
                "role": "user",
                "content": "" # need to fill before use inference
            }
        ],
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

  def tok_decode(self, tokens):
      return tokens

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
        try:
          # TODO(vvchernov): Use _model_generate_parallel(...) when it becomes possible
          self._model_generate_batch(request_batch, results)
        except ConnectionError as e:
          print(f"ConnectionError: {e}. Skipping this batch and continuing...")

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
        "Full time of predictions measurement: {:.2f} sec, {:.2f} min, {:.2f} hour(s)".format(
            secs, secs / 60, secs / 3600))

    return results

  def call_octoai_reset(self):
    try:
      resp = requests.post(self.url + "/chat/reset", headers = self.headers)
      return resp.json()
    except Exception as e:
      print(f"Error resetting chat for endpoint {self.url}")
      print(e)
      return

  def call_octoai_inference(self, user_input: str):
    self.data["messages"][0]["content"] = user_input
    response = requests.post(self.url + "/v1/chat/completions", headers=self.headers, json=self.data)

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
      if 'choices' in response.keys():
        success = True
        break
    if success:
      results.append(response['choices'][0]['message']['content'])
    else:
      print("ERROR: responce does not have choices. Dummy response was inserted")
      results.append("Dummy response")

  def _model_generate_batch(self, request_batch, results):
    parallel_results={}
    for id in range(len(request_batch)):
      parallel_results[id]=[]
      inp = request_batch[id][0]
      request_args = request_batch[id][1]
      until = request_args["until"]
      self._model_generate(inp, parallel_results[id], stop=until)

    # Collect results together
    for id in range(len(request_batch)):
      results.extend(parallel_results[id])

  def _model_generate_parallel(self, request_batch, results):
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=self.batch_size) as executor:
      futures = []
      parallel_results={}
      for id in range(len(request_batch)):
        parallel_results[id]=[]
        inp = request_batch[id][0]
        request_args = request_batch[id][1]
        until = request_args["until"]
        futures.append(executor.submit(self._model_generate, inp, parallel_results[id], stop=until))

      for future in concurrent.futures.as_completed(futures):
        try:
          future.result()
        except Exception as exc:
          print(f"Error parallel generating predictions: {exc}")

      # Collect results together
      for id in range(len(request_batch)):
        results.extend(parallel_results[id])
