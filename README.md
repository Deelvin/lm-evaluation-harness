# Language Model Evaluation Harness

## Overview

This is a Deelvin's fork of [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) made for development purposes. It contains a few features and workarounds for better evaluation and precision measurements.  
  
This repository is based on the revision used in [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
  
The main differences from the original repository:  
- Updated scorer for GSM8K and TriviaQA. Some models may not pick up the template of tasks with exact match metrics that can become a reason of significant gap between calculateed and actual metrics.  
- Additional backend for OctoAI models  
- Integration of [HumanEval](https://github.com/openai/human-eval) task  
- Advanced few-shot construction: use of model specific conversation template tags (e.g. `[INST]`, `[/INST]` for llama2) between few-shot samples

## Install

To install `lm-eval` from the github repository main branch, run:

```bash
git clone https://github.com/Deelvin/lm-evaluation-harness --branch=develop-hf
cd lm-evaluation-harness
pip install -e .
```

To install additional multilingual tokenization and text segmentation packages, you must install the package with the `multilingual` extra:

```bash
pip install -e ".[multilingual]"
```

To support loading GPTQ quantized models, install the package with the `auto-gptq` extra:

```bash
pip install -e ".[auto-gptq]"
```

## Basic Usage

### OctoAI

For OctoAI models you need to first export your token to environment variable `OCTOAI_TOKEN`. 

```bash
python main.py \
    --model octoai \
    --model_args model_name=mistral-7b-instruct,url=127.0.0.1:8000,batch_size=16 \
    --task gsm8k \
    --num_fewshot 5 \
    --conversation_template llama \
    --no_cache \
    --write_out
```
A few features of this fork are used here. `--model=octoai` backend supports the following args:  
`model_name`, `url`, `batch_size`, `temperature` and `top_p`.  
In the example above there is also `--conversation_template=llama` argument. This is a local feature which inserts tags from model dependant chat template between samples for few-shot inference. Usually it conviderably increases accuracy and improves the quality of model responses. **We highly recommend to use this option for architectures similar to llama (llama2, codellama, mistral-7b)**.  

>**Note**: There is one more feature implemented in the fork which can help to deal with surprisingly low precision for "generate until" tasks like GSM8K or TriviaQA: `--use_soft_scorer` option. Some models cannot adapt to pattern from few-shots and continue generating answers as it used to and it may cause the situation when the model actually generates many appropriate responses with correct answers, but the task scorer cannot extract it due to not following the answer pattern. The "soft scorers" introduced here can provide more gentle answers matching to give more plausible metrics. Based on our empirical analysis, it was revealed that **this option should be used with llama2 and codellama**.


### Hugging Face `transformers`

To evaluate a model hosted on the [HuggingFace Hub](https://huggingface.co/models) (e.g. GPT-J-6B) on `hellaswag` you can use the following command:


```bash
python main.py \
    --model hf-causal \
    --model_args pretrained=EleutherAI/gpt-j-6B \
    --tasks hellaswag \
    --device cuda:0
```

Additional arguments can be provided to the model constructor using the `--model_args` flag. Most notably, this supports the common practice of using the `revisions` feature on the Hub to store partially trained checkpoints, or to specify the datatype for running a model:

```bash
python main.py \
    --model hf-causal \
    --model_args pretrained=EleutherAI/pythia-160m,revision=step100000,dtype="float" \
    --tasks lambada_openai,hellaswag \
    --device cuda:0
```

To evaluate models that are loaded via `AutoSeq2SeqLM` in Huggingface, you instead use `hf-seq2seq`. *To evaluate (causal) models across multiple GPUs, use `--model hf-causal-experimental`*

> **Warning**: Choosing the wrong model may result in erroneous outputs despite not erroring.
