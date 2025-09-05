# Usage Guide

### <mark style="background-color:green;">Serving with 1 x H100/H200</mark>

1. Install SGLang following [the instruction](https://app.gitbook.com/s/FFtIWT8LEMaYiYzz0p8P/sglang-cookbook/installation/nvidia-h-series-a-series-and-rtx-gpus)
2. Serve the model

{% code overflow="wrap" %}
```bash
# gpt-oss-20b
python3 -m sglang.launch_server --model-path openai/gpt-oss-20b
```
{% endcode %}

{% code overflow="wrap" %}
```bash
# gpt-oss-120b
python3 -m sglang.launch_server --model-path openai/gpt-oss-120b --mem-fraction-static 0.95
```
{% endcode %}

### <mark style="background-color:green;">Serving with 2 x H100</mark>

1. Install SGLang following [the instruction](https://app.gitbook.com/s/FFtIWT8LEMaYiYzz0p8P/sglang-cookbook/installation/nvidia-h-series-a-series-and-rtx-gpus)
2. Serve the model

{% code overflow="wrap" %}
```bash
# gpt-oss-120b
python3 -m sglang.launch_server --model-path openai/gpt-oss-120b --tp 2
```
{% endcode %}

### <mark style="background-color:green;">Serving with 1 x B200</mark>

* Install SGLang following [the instruction](../installation/nvidia-blackwell-gpus.md)
* Serve the model

{% code overflow="wrap" %}
```bash
# gpt-oss-20b
python3 -m sglang.launch_server --model-path openai/gpt-oss-20b
```
{% endcode %}

{% code overflow="wrap" %}
```bash
# gpt-oss-120b
python3 -m sglang.launch_server --model-path openai/gpt-oss-120b
```
{% endcode %}

#### With Speculative Decoding

{% code overflow="wrap" %}
```bash
# On Hopper:
# - Tree decoding (topk > 1) and chain decoding (topk = 1) are supported on both FA3 and Triton backends.
# Example for topk = 1
python3 -m sglang.launch_server --model openai/gpt-oss-120b --speculative-algorithm EAGLE3 --speculative-draft-model-path lmsys/EAGLE3-gpt-oss-120b-bf16 --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --tp 4
# Example for topk > 1
python3 -m sglang.launch_server --model openai/gpt-oss-120b --speculative-algorithm EAGLE3 --speculative-draft-model-path lmsys/EAGLE3-gpt-oss-120b-bf16 --speculative-num-steps 5 --speculative-eagle-topk 4 --speculative-num-draft-tokens 8 --tp 4

# On Blackwell:
# - Chain decoding (topk = 1) is supported on TRTLLM-MHA backend. Tree decoding (topk > 1) is in progress, stay tuned!
# - Both tree decoding (topk > 1) and chain decoding (topk = 1) are supported on the Triton backend.
# Example for topk = 1
python3 -m sglang.launch_server --model openai/gpt-oss-120b --speculative-algo EAGLE3 --speculative-draft lmsys/EAGLE3-gpt-oss-120b-bf16 --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --tp 4
# Example for topk > 1
python3 -m sglang.launch_server --model openai/gpt-oss-120b --speculative-algo EAGLE3 --speculative-draft lmsys/EAGLE3-gpt-oss-120b-bf16 --speculative-num-steps 5 --speculative-eagle-topk 4 --speculative-num-draft-tokens 8 --attention-backend triton --tp 4
```
{% endcode %}

### Responses API & Built-in Tools

Please refer to https://docs.sglang.ai/basic\_usage/gpt\_oss.html#responses-api-built-in-tools

### Benchmark & Accuracy Test

#### Benchmark Command

```bash
pip install tabulate

# Batch size 1
python3 -m sglang.bench_one_batch_server --model openai/gpt-oss-120b --base-url http://localhost:30000 --batch-size 1 --input-len 1024 --output-len 512

# Batch size 32
python3 -m sglang.bench_one_batch_server --model openai/gpt-oss-120b --base-url http://localhost:30000 --batch-size 32 --input-len 1024 8192 --output-len 512 --show-report
```

#### Test Accuracy

1. Install gpt-oss

```bash
git clone https://github.com/openai/gpt-oss.git
cd gpt-oss
pip install -e .
```

2. Evaluation Command

> You can choose a reasoning effort level from `low, medium, high`

```bash
OPENAI_API_KEY=dummy python -m gpt_oss.evals \\
    --base-url http://localhost:30000/v1 \\
    --model dummy \\
    --reasoning-effort medium \\
    --eval gpqa \\
    --n-threads 1000
```

### Known Limitations

* On Blackwell, the default attention backend (`trtllm_mha`) does not support `topk > 1` when speculative decoding is enabled. To use `topk > 1`, you should switch to a different attention backend (e.g., Triton) by specifying `--attention-backend triton`
