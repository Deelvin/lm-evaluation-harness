import os
import json
from typing import List, Tuple, Union, Iterable
from lm_eval.base import BaseLM

import torch
import numpy as np
from transformers import AutoTokenizer

import tvm
from tvm import relax


def load_params(params_path: str, device) -> List[tvm.nd.NDArray]:
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
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_batch_size = max_batch_size
        self.device = device
        self.model_path = os.Path(model_path)

        self.params_path = self.model_path.joinpath("params")
        self.config_path = self.params_path

        self.mlc_config = {}
        with open(self.config_path) as file:
            self.mlc_config = json.load(file)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.mlc_config["tokenizer_files"][0], 
            trust_remote_code=True
        )

        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        if self.model_name.startswith("dolly-"):
            # 50277 means "### End"
            self.tokenizer.eos_token_id = 50277

        self.const_params = load_params(self.params_path, self.device)
        ex = tvm.runtime.load_module(
            os.path.join(
                model_path,
                f"{model_name}-{device}.so",
            )
        )
        
        self.vm = relax.VirtualMachine(ex, self.device)

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
        return self.batch_size

    @property
    def device(self):
        return self.device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string)

    def tok_decode(self, tokens: Iterable[int]):
        return self.tokenizer.batch_decode(tokens)


    def _sample_top_p(self, probs, p):
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > p
        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(probs_sort, num_samples=1)
        next_token = torch.gather(probs_idx, -1, next_token)

        return next_token

    def _model_call(
        self, 
        inps: torch.Tensor, 
        seq_len: int,
        reset: bool = False
    ) -> torch.Tensor:
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
        max_length: int
    ) -> torch.Tensor:
        prompt_len = context.shape[1]
        total_len = max_length + prompt_len
        tvm_tokens = tvm.nd.array(
            np.zeros(
                (1, total_len), 
                dtype="int32"
            ), 
            device=self.device
        )
        tokens = torch.from_dlpack(tvm_tokens)
        tokens[0, : prompt_len] = context
        start_pos = prompt_len
        for cur_pos in range(start_pos, total_len):
            if cur_pos == start_pos:
                logits = self._model_call(tokens[:, :cur_pos], cur_pos, reset=True)
            else:
                tvm_to_model = tvm.nd.array(
                    np.zeros(
                        (1, 1), 
                        dtype="int32"
                    ), 
                    device=self.device
                )
                to_model = torch.from_dlpack(tvm_to_model)
                to_model[0, 0] = tokens[:, cur_pos - 1 : cur_pos]
                logits = self.model(to_model)
            logits = logits[:, -1, :].to(torch.float32)
            if self.mlc_config["temperature"] > 0:
                probs = torch.softmax(logits / self.mlc_config["temperature"], dim=-1)
                next_token = self._sample_top_p(probs, self.mlc_config["top_p"])
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            tokens[:, cur_pos] = next_token

            if next_token[0] in self.stop_tokens:
                break

        return tokens[:, :cur_pos + 1]

    def greedy_until(
        self, 
        requests: List[Tuple[str, Union[List[str], str]]]
    ) -> List[str]:
        if not requests:
            return []

        results = []

        for request in requests:
            inp = request[0]
            request_args = request[1]
            until = request_args["until"]
            results.append(self._model_generate(inp, results, stop=until))

        return results