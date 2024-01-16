import os
from typing import List, Tuple, Union, Iterable, Optional

import torch

from lm_eval.base import BaseLM
from transformers import AutoTokenizer

class TRTLM(BaseLM):
    def __init__(
        self,
        model_name: str,
        engine_path: str,
        tokenizer_path: str,
        batch_size: int = 1,
    ):
        from tensorrt_llm.runtime import ModelRunner
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            legacy=False,
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
            use_fast=True
        )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = ModelRunner.from_dir(
            engine_dir=engine_path
        )
        self.model_name = model_name

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
        raise "cuda"

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _loglikelihood_tokens(self, requests):
        raise NotImplementedError()

    def _model_call(self, inps):
        raise NotImplementedError()

    def _model_generate(
        self, 
        context: torch.Tensor,
        max_length: int,
        eos_token_id: Optional[List[str]] = None
    ) -> torch.Tensor:
        # TODO: support batch generation
        with torch.no_grad():
            outputs = self.model.generate(
                [context.unsqueeze(dim=1)],
                max_new_tokens=self.max_gen_toks,
                end_id=self.tokenizer.eos_token_id,
                pad_id=self.tokenizer.pad_token_id,
                temperature=0.0,
                repetition_penalty=0.0,
            )
            torch.cuda.synchronize()
        return outputs.squeeze()

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
        return res
