import os
import json
from typing import List, Tuple, Union, Iterable, Optional

import torch
import numpy as np
from transformers import AutoTokenizer

from lm_eval.base import BaseLM


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
        rpc_host: str = None,
        rpc_port: int = None,
        rpc_key: str = None,
        so_name: str = None,
    ):
        import tvm  # pylint: disable=import-outside-toplevel
        from tvm import relax  # pylint: disable=import-outside-toplevel
        from tvm import rpc

        super().__init__()

        self.model_name = model_name
        self._batch_size = int(batch_size)
        self.max_batch_size = max_batch_size
        self._device = device
        self._tvm_device = tvm.device(self._device)
        self.model_path = model_path

        self.params_path = os.path.join(self.model_path, "params")
        self.config_path = os.path.join(self.params_path, "mlc-chat-config.json")
        if not os.path.isfile(self.config_path):
            self.params_path = self.model_path
            self.config_path = os.path.join(self.params_path, "mlc-chat-config.json")

        if not os.path.isfile(self.config_path):
            paths = [
                os.path.join(self.model_path, "params"),
                self.model_path,
            ]
            raise Exception(
                "Cannot find mlc-chat-config.json. Looked at: {}".format(paths)
            )

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

        self.remote = None
        if rpc_host is not None and rpc_port is not None and rpc_key is not None:
            tracker = rpc.connect_tracker(rpc_host, rpc_port)
            self.remote = tracker.request(rpc_key, priority=0)

        if so_name is None:
            so_name = f"{model_name}-{self._device}.so"
        so_path = os.path.join(
            model_path,
            so_name,
        )

        self.tot_seq_len = 0
        if self.remote is None:
            self.const_params = load_params(self.params_path, self._tvm_device)
            ex = tvm.runtime.load_module(so_path)
            self.vm = relax.VirtualMachine(ex, self._tvm_device)
            self.kv_cache = self.vm["create_kv_cache"]()
            self.kv_cache_clear = tvm.get_global_func(
                "vm.builtin.attention_kv_cache_array_clear"
            )
        else:
            mod_rpc = self.remote.get_function("mlc.llm_chat_create")(
                self._tvm_device.device_type, self._tvm_device.device_id
            )
            self.vm = {
                "prefill": mod_rpc.get_function("prefill"),
                "decode": mod_rpc.get_function("decode"),
                "stopped": mod_rpc.get_function("stopped"),
                "get_message": mod_rpc.get_function("get_message"),
                "reload": mod_rpc.get_function("reload"),
                "unload": mod_rpc.get_function("unload"),
                "evaluate": mod_rpc.get_function("evaluate"),
                "raw_generate": mod_rpc.get_function("raw_generate"),
                "embed": mod_rpc.get_function("embed"),
                "prefill_with_embed": mod_rpc.get_function("prefill_with_embed"),
                "load_json_override": mod_rpc.get_function("load_json_override"),
                "get_role0": mod_rpc.get_function("get_role0"),
                "get_role1": mod_rpc.get_function("get_role1"),
                "runtime_stats_text": mod_rpc.get_function("runtime_stats_text"),
                "verbose_runtime_stats_text": mod_rpc.get_function(
                    "verbose_runtime_stats_text"
                ),
                "reset_chat": mod_rpc.get_function("reset_chat"),
                "reset_runtime_stats": mod_rpc.get_function("reset_runtime_stats"),
                "get_config_json": mod_rpc.get_function("get_config_json"),
                "process_system_prompts": mod_rpc.get_function(
                    "process_system_prompts"
                ),
            }
            # remote.upload(so_path)
            # rlib = remote.load_module(so_name)
            params_path = "/data/local/tmp/Llama-2-7b-chat-hf-q4f16_1-MLC"
            params_path = "/media/deelvin_disk/echuraev/Workspace/OctoML/mlc-llm/dist/Llama-2-7b-chat-hf-q4f16_1-MLC"
            self.vm["reload"](so_name, params_path)

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
        return self.tokenizer.batch_decode(tokens)[0]  # for single sentence

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
        self, inps: torch.Tensor, seq_len: int = 1, reset: bool = False
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
        eos_token_id: Optional[List[str]] = None,
    ) -> torch.Tensor:
        import tvm  # pylint: disable=import-outside-toplevel

        prompt_len = context.shape[0]
        total_len = max_length + prompt_len
        tvm_tokens = tvm.nd.array(
            np.zeros((1, total_len), dtype="int32"), device=self._tvm_device
        )
        tokens = torch.from_dlpack(tvm_tokens)
        tokens[0, :prompt_len] = context
        start_pos = prompt_len
        eos_token_id = torch.tensor(self.tokenizer.eos_token_id).to(self._device)
        for cur_pos in range(start_pos, total_len):
            if cur_pos == start_pos:
                logits = self._model_call(tokens[:, :cur_pos], cur_pos, reset=True)
            else:
                tvm_to_model = tvm.nd.array(
                    np.zeros((1, 1), dtype="int32"), device=self._tvm_device
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
        self, requests: List[Tuple[str, Union[List[str], str]]]
    ) -> List[str]:
        if not requests:
            return []

        results = []

        for num, request in enumerate(requests):
            inp = request[0]
            request_args = request[1]
            until = request_args["until"]
            if self.remote is None:
                results.append(
                    self.tok_decode(
                        self._model_generate(
                            torch.Tensor(self.tok_encode(inp)),
                            self.max_length,
                            eos_token_id=until,
                        )
                    )
                )
            else:
                self.vm["reset_chat"]()
                self.vm["process_system_prompts"]()
                self.vm["prefill"](str(inp))
                while self.vm["stopped"]() == 0:
                    self.vm["decode"]()
                output = self.vm["get_message"]()
                results.append(output)
            print(f"\r{num}/{len(requests)} requests processed", end="")

        return results
