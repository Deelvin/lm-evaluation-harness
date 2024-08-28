import os
from functools import cached_property
from typing import Dict, List, Optional, Tuple, Union

from lm_eval.api.registry import register_model
from lm_eval.models.api_models import TemplateAPI
from lm_eval.utils import eval_logger


@register_model(
    "octoai-completions",
)
class OctoAICompletionsAPI(TemplateAPI):
    def __init__(
        self,
        base_url="https://text.octoai.run/v1/completions",
        tokenizer_backend="huggingface",
        tokenized_requests=False,
        **kwargs,
    ):
        super().__init__(
            base_url=base_url,
            tokenizer_backend=tokenizer_backend,
            tokenized_requests=tokenized_requests,
            **kwargs,
        )

    @cached_property
    def api_key(self):
        """Override this property to return the API key for the API request."""
        key = os.environ.get("OCTOAI_TOKEN", None)
        if key is None:
            raise ValueError(
                "API key not found. Please set the OCTOAI_TOKEN environment variable."
            )
        return key

    def _create_payload(
        self,
        messages: Union[List[List[int]], List[dict], List[str], str],
        generate=False,
        gen_kwargs: Optional[dict] = None,
        **kwargs,
    ) -> dict:
        if generate:
            gen_kwargs.pop("do_sample", False)
            max_tokens = gen_kwargs.pop("max_gen_toks", self._max_gen_toks)
            temperature = gen_kwargs.pop("temperature", 0)
            stop = gen_kwargs.pop("until", None)
            return {
                "prompt": messages,
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stop": stop,
                **gen_kwargs,
            }
        else:
            return {
                "model": self.model,
                "prompt": messages,
                "temperature": 0,
                "max_tokens": 1,
                "logprobs": 1,
                "loglikelihood": True,
            }

    # @staticmethod
    def parse_logprobs(
        self,
        outputs: Union[Dict, List[Dict]],
        tokens: List[List[int]] = None,
        ctxlens: List[int] = None,
        **kwargs,
    ) -> List[Tuple[float, bool]]:
        res = []
        if not isinstance(outputs, list):
            outputs = [outputs]
        for out in outputs:
            for choice, ctxlen in zip(out["choices"], ctxlens):
                assert ctxlen > 0, "Context length must be greater than 0"
                logprobs = sum(
                    [content["logprob"] for content in choice["logprobs"]["content"]][
                        ctxlen:
                    ]
                )
                tokens = [
                    content["token"] for content in choice["logprobs"]["content"]
                ][ctxlen:]
                top_logprobs = [
                    content["top_logprobs"] for content in choice["logprobs"]["content"]
                ][ctxlen:]
                is_greedy = True
                for tok, top in zip(tokens, top_logprobs):
                    tok_ids = self.tokenizer.encode(tok, add_special_tokens=False)
                    space_token = self.tokenizer.convert_tokens_to_ids("_")
                    top_ids = self.tokenizer.encode(
                        top[0]["token"], add_special_tokens=False
                    )
                    if space_token in top_ids and space_token not in tok_ids:
                        top_ids = top_ids[1:]
                    if len(tok_ids) != len(top_ids) or not all(
                        [tok_id == top_id for tok_id, top_id in zip(tok_ids, top_ids)]
                    ):
                        is_greedy = False
                        break
                res.append((logprobs, is_greedy))
        return res

    @staticmethod
    def parse_generations(outputs: Union[Dict, List[Dict]], **kwargs) -> List[str]:
        res = []
        if not isinstance(outputs, list):
            outputs = [outputs]
        for out in outputs:
            for choices in out["choices"]:
                res.append(choices["text"])
        return res


@register_model("octoai-chat-completions")
class OctoAIChatCompletion(OctoAICompletionsAPI):
    def __init__(
        self,
        base_url="https://text.octoai.run/v1/chat/completions",
        **kwargs,
    ):
        eval_logger.warning(
            "chat-completions endpoint requires the `--apply_chat_template` flag."
        )
        super().__init__(
            base_url=base_url,
            **kwargs,
        )

    @cached_property
    def api_key(self):
        """Override this property to return the API key for the API request."""
        key = os.environ.get("OCTOAI_TOKEN", None)
        if key is None:
            raise ValueError(
                "API key not found. Please set the OCTOAI_TOKEN environment variable."
            )
        return key

    def _create_payload(
        self,
        messages: Union[List[Dict], str],
        generate=False,
        gen_kwargs: dict = {},
        **kwargs,
    ) -> dict:
        if generate:
            gen_kwargs.pop("do_sample", False)
            max_tokens = gen_kwargs.pop("max_gen_toks", self._max_gen_toks)
            temperature = gen_kwargs.pop("temperature", 0)
            stop = gen_kwargs.pop("until", None)
            if stop and not isinstance(stop, (list, tuple)):
                stop = [stop]

            return {
                "messages": messages,
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stop": stop,
                **gen_kwargs,
            }
        else:
            return {
                "messages": messages,
                "model": self.model,
                "temperature": 0,
                "max_tokens": 1,
                "logprobs": 1,
                "loglikelihood": True,
            }

    @staticmethod
    def parse_generations(outputs: Union[Dict, List[Dict]], **kwargs) -> List[str]:
        res = []
        if not isinstance(outputs, list):
            outputs = [outputs]
        for out in outputs:
            for choices in out["choices"]:
                res.append(choices["message"]["content"])
        return res

    def create_message(
        self,
        messages: Union[str, Tuple[str]],
    ) -> List[Dict[str, str]]:
        """Helper method to transform the prompt into the expected API input format. messages consist of batched requests"""
        if isinstance(messages, str):
            prompt = [{"role": "user", "content": f"{messages}"}]
        elif isinstance(messages, (Tuple, list)):
            prompt = messages[0]
        return [{"role": "user", "content": f"{prompt}"}]
