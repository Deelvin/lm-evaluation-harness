import os
import json
from typing import List, Tuple, Union, Iterable, Optional

import requests
import torch
import numpy as np
from transformers import AutoTokenizer

import asyncio
from lm_eval.base import BaseLM
import aiohttp
REPEAT_REQUEST_TO_MLCSERVE_SERVER = 10

def load_params(params_path: str, device):
    from tvm.contrib import tvmjs  # pylint: disable=import-outside-toplevel

    params, meta = tvmjs.load_ndarray_cache(f"{params_path}", device)
    plist = []
    size = meta["ParamSize"]
    for i in range(size):
        plist.append(params[f"param_{i}"])
    return plist


class MLCLM(BaseLM):
    def __init__(
        self,
        model_name: str,
        model_path: str,
        batch_size: int = 1,
        max_batch_size: int = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        temperature: float = 0.0,
        top_p: float = 0.0,
    ):
        import tvm  # pylint: disable=import-outside-toplevel
        from tvm import relax # pylint: disable=import-outside-toplevel
        super().__init__()

        self.model_name = model_name
        self._batch_size = int(batch_size)
        self.max_batch_size = max_batch_size
        self._device = device
        self._tvm_device = tvm.device(self._device)
        self.model_path = model_path

        self.params_path = os.path.join(self.model_path, "params")
        self.config_path = os.path.join(self.params_path, "mlc-chat-config.json")

        self.mlc_config = {}
        with open(self.config_path) as file:
            self.mlc_config = json.load(file)

        if temperature is not None:
            self.mlc_config["temperature"] = temperature
        if top_p is not None:
            self.mlc_config["top_p"] = top_p

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.params_path,
            trust_remote_code=True,
        )

        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        if self.model_name.startswith("dolly-"):
            # 50277 means "### End"
            self.tokenizer.eos_token_id = 50277

        self.const_params = load_params(self.params_path, self._tvm_device)
        ex = tvm.runtime.load_module(
            os.path.join(
                model_path,
                f"{model_name}-{self._device}.so",
            )
        )

        self.vm = relax.VirtualMachine(ex, self._tvm_device)

        self.tot_seq_len = 0
        self.kv_cache = self.vm["create_kv_cache"]()
        self.kv_cache_clear = tvm.get_global_func("vm.builtin.attention_kv_cache_array_clear")

        try:
            self.prefill_func = self.vm["prefill"]
        except AttributeError:
            self.prefill_func = None

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

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
        return self.tokenizer.encode(string)

    def tok_decode(self, tokens: Iterable[int]):
        return self.tokenizer.batch_decode(tokens)[0] # for single sentence


    def _sample_top_p(self, probs, p):
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > p
        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(probs_sort, num_samples=1)
        next_token = torch.gather(probs_idx, -1, next_token)

        return next_token

    def reset(self):
        self.kv_cache_clear(self.kv_cache)
        self.tot_seq_len = 0

    def _model_call(
        self,
        inps: torch.Tensor,
        seq_len: int = 1,
        reset: bool = False
    ) -> torch.Tensor:
        import tvm  # pylint: disable=import-outside-toplevel
        if reset:
            self.reset()
        self.tot_seq_len += seq_len
        seq_len_shape = tvm.runtime.ShapeTuple([self.tot_seq_len])
        if seq_len > 1 and self.prefill_func:
            inps = tvm.nd.from_dlpack(inps)
            logits, kv_cache = self.prefill_func(
                inps, seq_len_shape, self.kv_cache, self.const_params
            )
        else:
            for i in range(seq_len):
                input_slice = tvm.nd.from_dlpack(inps[:, i : i + 1])
                logits, kv_cache = self.vm["decode"](
                    input_slice, seq_len_shape, self.kv_cache, self.const_params
                )
        self.kv_cache = kv_cache

        return torch.from_dlpack(logits)

    def _model_generate(
        self,
        context: torch.Tensor,
        max_length: int,
        eos_token_id: Optional[List[str]] = None
    ) -> torch.Tensor:
        import tvm  # pylint: disable=import-outside-toplevel
        prompt_len = context.shape[0]
        total_len = max_length + prompt_len
        tvm_tokens = tvm.nd.array(
            np.zeros(
                (1, total_len),
                dtype="int32"
            ),
            device=self._tvm_device
        )
        tokens = torch.from_dlpack(tvm_tokens)
        tokens[0, : prompt_len] = context
        start_pos = prompt_len
        eos_token_id = torch.tensor(self.tokenizer.eos_token_id).to(self._device)
        for cur_pos in range(start_pos, total_len):
            if cur_pos == start_pos:
                logits = self._model_call(tokens[:, :cur_pos], cur_pos, reset=True)
            else:
                tvm_to_model = tvm.nd.array(
                    np.zeros(
                        (1, 1),
                        dtype="int32"
                    ),
                    device=self._tvm_device
                )
                to_model = torch.from_dlpack(tvm_to_model)
                to_model[0, 0] = tokens[:, cur_pos - 1 : cur_pos]
                logits = self._model_call(to_model)
            logits = logits[:, -1, :].to(torch.float32)
            if self.mlc_config["temperature"] > 0:
                probs = torch.softmax(logits / self.mlc_config["temperature"], dim=-1)
                next_token = self._sample_top_p(probs, self.mlc_config["top_p"])
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            tokens[:, cur_pos] = next_token

            if next_token[0] == eos_token_id:
                break

        return tokens[:, start_pos : cur_pos + 1]

    def greedy_until(
        self,
        requests: List[Tuple[str, Union[List[str], str]]]
    ) -> List[str]:
        if not requests:
            return []

        results = []

        for num, request in enumerate(requests):
            inp = request[0]
            request_args = request[1]
            until = request_args["until"]
            results.append(
                self.tok_decode(
                    self._model_generate(
                        torch.Tensor(self.tok_encode(inp)),
                        self.max_length,
                        eos_token_id=until
                    )
                )
            )
            print(f"\r{num}/{len(requests)} requests processed", end="")

        return results


class MLCServe(BaseLM):
    def __init__(
        self,
        model_name: str,
        ip: str = "0.0.0.0",
        port: int = 32777,
        batch_size: int = 1,
        temperature: float = 0.0,
        top_p: float = 1.0,
        parallel: bool = False,
    ):
        super().__init__()

        self.model_name = model_name
        self.ip = ip
        self.port = port
        self._batch_size = int(batch_size)
        self.temperature = temperature
        self.top_p = top_p

        self.url_suffix = "/v1/chat/completions"
        self.parallel = parallel or self._batch_size > 1
        assert (
            (self._batch_size == 1 and not self.parallel) or
            (self._batch_size > 1 and self.parallel)
        ), "Please insert batch size bigger than 1 for parallel regime"

        self.headers = {
            "Content-Type": "application/json",
        }

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

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

    def tok_decode(self, tokens: Iterable[int]):
        raise NotImplementedError("Cannot call tokenizer by API")

    def create_chat_completion_payload(
        self,
        prompt,
        loglikelihood = False,
        stop_tokens = None,
    ):
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "stream": False,
            "stop": stop_tokens,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "loglikelihood": loglikelihood,
        }

        return payload

    def prepare_msg(self, request, loglikelihood=False):
        prompt = request[0]
        request_args = request[1]
        until = request_args["until"]

        return self.create_chat_completion_payload(prompt, loglikelihood, until)

    async def send_request(self, payload):
        success = False
        async with aiohttp.ClientSession() as session:
            for i in range(REPEAT_REQUEST_TO_MLCSERVE_SERVER):
                async with session.post(
                    f"http://{self.ip}:{self.port}{self.url_suffix}",
                    headers=self.headers,
                    json=payload,
                ) as response:
                    response_json = await response.json()
                    if response.status == 200:
                        success = True
                        break
                    else:
                        print(f"Error (iteration {i}): status = {response.status}\nJson:\n{response_json}")
        if success:
            return response_json
        else:
            print("ERROR: request sending failed. Dummy response was inserted")
            return None

    def get_output(self, response):
        if response is None:
            return self.dummy_output()
        return response["choices"][0]["message"]["content"]

    def dummy_output(self):
        return "Dummy response"

    async def model_call(self, request, results):
        payload = self.prepare_msg(request)
        output_json = await self.send_request(payload)

        results.append(
            self.get_output(output_json)
        )

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

    def get_output_loglikelihood(self, response, context, continuation):
        if response is None:
            return self.dummy_loglikelihood_output()

        logprob_content = response["choices"][0]["logprobs"]["content"]
        logprobs = []
        tokens = []
        top1_tokens = []
        for content in logprob_content:
            tokens.append(content["token"])
            logprobs.append(content["logprob"])
            top1_tokens.append(content["top_logprobs"][0]["token"])

        # Calculate context length
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
            return self.dummy_loglikelihood_output()

        res_logprob = sum(logprobs[-cont_len:])
        tokens_len = len(tokens)
        res_is_greedy = True
        for i in range(tokens_len - cont_len, tokens_len):
            if top1_tokens[i] != tokens[i]:
                res_is_greedy = False
                break
        return res_logprob, res_is_greedy

    def dummy_loglikelihood_output(self):
        import sys
        return -sys.float_info.max, False

    async def model_call_loglikelihood(self, request, results):
        payload = self.prepare_msg(
            request = (
                request[0] + request[1],
                {"until": []},
            ),
            loglikelihood = True,
        )
        output_json = await self.send_request(payload)

        results.append(
            self.get_output_loglikelihood(output_json, request[0], request[1])
        )

    async def model_generate_parallel(self, request_batch, results, loglikelihood=False):
        parallel_results = {}
        for id in range(len(request_batch)):
            parallel_results[id]=[]

        if loglikelihood:
            run = self.model_call_loglikelihood
        else:
            run = self.model_call

        tasks = [run(request_batch[id], parallel_results[id]) for id in range(len(request_batch))]
        await asyncio.gather(*tasks)
        for id in range(len(request_batch)):
            results.extend(parallel_results[id])

    def _batcher(self, requests):
        for i in range(0, len(requests), self._batch_size):
            yield requests[i:i + self._batch_size]

    def parallel_requests(self, requests, results, loglikelihood=False):
        for batch_idx, request_batch in enumerate(self._batcher(requests)):
            try:
                asyncio.run(self.model_generate_parallel(request_batch, results, loglikelihood=loglikelihood))
            except ConnectionError as e:
                print(f"ConnectionError: {e}. Skipping this batch and continuing...")
                print(
                    f"\r{(batch_idx + 1) * self._batch_size}/{len(requests)} requests processed",
                    end="",
                )

    def greedy_until(
        self,
        requests: List[Tuple[str, Union[List[str], str]]]
    ) -> List[str]:
        if not requests:
            return []

        results = []
        if self.parallel:
            self.parallel_requests(requests, results)
        else:
            for num, request in enumerate(requests):
                asyncio.run(self.model_call(request, results))
                print(f"\r{num}/{len(requests)} requests processed", end="")

        return results

    def loglikelihood(
        self,
        requests: List[Tuple[str, str]],
    ) -> List[Tuple[float, bool]]:
        if not requests:
            return []

        results = []
        if self.parallel:
            self.parallel_requests(requests, results, loglikelihood=True)
        else:
            for num, request in enumerate(requests):
                asyncio.run(self.model_call_loglikelihood(request, results))
                print(f"\r{num}/{len(requests)} requests processed", end="")

        return results

    def _model_call(self, inps):
        raise NotImplementedError("MLC-LLM server does not support one model call in current format, loglikelyhood method will be overrided")

    def _model_generate(self, context, max_length, eos_token_id):
        raise NotImplementedError("MLC-LLM server does not support model generate in current format, greedy_until method was override")
