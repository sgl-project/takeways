# Usage Guide

### <mark style="background-color:green;">Serving with 1 x H100/H200</mark>

{% stepper %}
{% step %}
### Install SGLang

Following [the instruction](https://app.gitbook.com/o/TvLfyTxdRQeudJH7e5QW/s/FFtIWT8LEMaYiYzz0p8P/~/changes/11/sglang-cookbook/installation/nvidia-h-series-a-series-and-rtx-gpus)
{% endstep %}

{% step %}
### Serve the model

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
{% endstep %}

{% step %}
### Benchmark

SGLang version (0.5.1)

<pre class="language-bash" data-overflow="wrap"><code class="lang-bash"><strong># gpt-oss-20b
</strong>python -m sglang.bench_one_batch_server --base-url http://127.0.0.1:30000  --model-path openai/gpt-oss-20b --batch 1 --input-len 1024 --output-len 1024 
</code></pre>

<table><thead><tr><th width="209.78515625">BS/Input/Output Length</th><th width="109.6328125">TTFT(s)</th><th width="101.75390625">ITL(ms)</th><th>Input Throughput</th><th>Output Throughput</th></tr></thead><tbody><tr><td>1/1024/1024</td><td>0.05</td><td>3.29</td><td>22668.19</td><td>304.59</td></tr><tr><td>1/8192/1024</td><td>0.15</td><td>3.39</td><td>55870.90</td><td>295.09</td></tr><tr><td>8/1024/1024</td><td>0.12</td><td>5.92</td><td>65760.01</td><td>1350.83</td></tr><tr><td>8/8192/1024</td><td>1.05</td><td>6.62</td><td>62209.72</td><td>1209.10</td></tr></tbody></table>

<pre class="language-bash" data-overflow="wrap"><code class="lang-bash"><strong># gpt-oss-120b
</strong>python -m sglang.bench_one_batch_server --base-url http://127.0.0.1:30000  --model-path openai/gpt-oss-120b --batch 1 --input-len 1024 --output-len 1024 
</code></pre>

<table><thead><tr><th width="209.78515625">BS/Input/Output Length</th><th width="109.6328125">TTFT(s)</th><th width="101.75390625">ITL(ms)</th><th>Input Throughput</th><th>Output Throughput</th></tr></thead><tbody><tr><td>1/1024/1024</td><td>0.07</td><td>4.73</td><td>15803.59</td><td>211.49</td></tr><tr><td>1/8192/1024</td><td>0.23</td><td>4.89</td><td>35004.05</td><td>204.75</td></tr><tr><td>8/1024/1024</td><td>0.21</td><td>10.17</td><td>39132.98</td><td>786.63</td></tr><tr><td>8/8192/1024</td><td>1.76</td><td>11.20</td><td>37178.23</td><td>714.53</td></tr></tbody></table>
{% endstep %}
{% endstepper %}

### <mark style="background-color:green;">Serving with 2 x H100</mark>

{% stepper %}
{% step %}
### Install SGLang

Following [the instruction](https://app.gitbook.com/o/TvLfyTxdRQeudJH7e5QW/s/FFtIWT8LEMaYiYzz0p8P/~/changes/11/sglang-cookbook/installation/nvidia-h-series-a-series-and-rtx-gpus)
{% endstep %}

{% step %}
### Serve the model

{% code overflow="wrap" %}
```bash
# gpt-oss-120b
python3 -m sglang.launch_server --model-path openai/gpt-oss-120b --tp 2
```
{% endcode %}
{% endstep %}

{% step %}
### Benchmark

<table><thead><tr><th width="209.78515625">BS/Input/Output Length</th><th width="109.6328125">TTFT(s)</th><th width="101.75390625">ITL(ms)</th><th>Input Throughput</th><th>Output Throughput</th></tr></thead><tbody><tr><td colspan="5" style="text-align: center;">Benchmark results will be added here</td></tr></tbody></table>
{% endstep %}
{% endstepper %}

### <mark style="background-color:green;">Serving with 1 x B200</mark>

{% stepper %}
{% step %}
### Install SGLang

Following [the instruction](../installation/nvidia-blackwell-gpus.md)
{% endstep %}

{% step %}
### Serve the model

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
{% endstep %}

{% step %}
### With Speculative Decoding

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
{% endstep %}

{% step %}
### Benchmark

<table><thead><tr><th width="209.78515625">BS/Input/Output Length</th><th width="109.6328125">TTFT(s)</th><th width="101.75390625">ITL(ms)</th><th>Input Throughput</th><th>Output Throughput</th></tr></thead><tbody><tr><td colspan="5" style="text-align: center;">Benchmark results will be added here</td></tr></tbody></table>
{% endstep %}
{% endstepper %}

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
