# Usage Guide

### <mark style="background-color:green;">Serving with 1 x 8 x H200</mark>

1.  Install SGLang following [the instruction](https://app.gitbook.com/s/FFtIWT8LEMaYiYzz0p8P/sglang-cookbook/installation/nvidia-h-series-a-series-and-rtx-gpus)

    Note if you are using RDMA and are using docker, `--network host` and `--privileged` are required for `docker run` command.
2. Serve the model

{% code overflow="wrap" %}
```bash
python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-V3 --tp 8 --trust-remote-code --port 30000
```
{% endcode %}

* You may need to set `NCCL_IB_GID_INDEX` if you are using RoCE, for example: `export NCCL_IB_GID_INDEX=3`.
* [Optional Optimization Options](./#optional-performance-optimization)

### <mark style="background-color:green;">Serving with 1 x 8 x MI300X</mark>

1. Install SGLang following [the instruction](../installation/amd-gpus.md)
2. Serve the model

{% code overflow="wrap" %}
```bash
python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-V3 --tp 8 --trust-remote-code --port 30000
```
{% endcode %}

[Running DeepSeek-R1 on a single NDv5 MI300X VM](https://techcommunity.microsoft.com/blog/azurehighperformancecomputingblog/running-deepseek-r1-on-a-single-ndv5-mi300x-vm/4372726) could also be a good reference.

### <mark style="background-color:green;">Serving with 2 x 8 x H100/800/20</mark>

1. Install SGLang following [the instruction](https://app.gitbook.com/s/FFtIWT8LEMaYiYzz0p8P/sglang-cookbook/installation/nvidia-h-series-a-series-and-rtx-gpus) for the 2 nodes
2. Serve the model

If the first node's IP is `10.0.0.1` , launch the server in both node with below commands

{% code overflow="wrap" %}
```bash
# node 1
python3 -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V3 --tp 16 --dist-init-addr 10.0.0.1:5000 --nnodes 2 --node-rank 0 --trust-remote-code

# node 2
python3 -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V3 --tp 16 --dist-init-addr 10.0.0.1:5000 --nnodes 2 --node-rank 1 --trust-remote-code
```
{% endcode %}

* If the command fails, try setting the `GLOO_SOCKET_IFNAME` parameter. For more information, see [Common Environment Variables](https://pytorch.org/docs/stable/distributed.html#common-environment-variables).
* If the multi nodes support NVIDIA InfiniBand and encounter hanging issues during startup, consider adding the parameter `export NCCL_IB_GID_INDEX=3`. For more information, see [this](https://github.com/sgl-project/sglang/issues/3516#issuecomment-2668493307).
* [Optional Optimization Options](./#optional-performance-optimization)

### <mark style="background-color:green;">Serving with Xeon 6980P CPU</mark>

1. Install SGLang following [the instruction](../installation/intel-xeon-cpus.md)
2. Serve the model

* For w8a8\_int8

```bash
python -m sglang.launch_server                 \
    --model meituan/DeepSeek-R1-Channel-INT8   \
    --trust-remote-code                        \
    --disable-overlap-schedule                 \
    --device cpu                               \
    --quantization w8a8_int8                   \
    --host 0.0.0.0                             \
    --mem-fraction-static 0.8                  \
    --max-total-token 65536                    \
    --tp 6
```

* For FP8

```bash
python -m sglang.launch_server                 \
    --model deepseek-ai/DeepSeek-R1            \
    --trust-remote-code                        \
    --disable-overlap-schedule                 \
    --device cpu                               \
    --host 0.0.0.0                             \
    --mem-fraction-static 0.8                  \
    --max-total-token 65536                    \
    --tp 6
```

### <mark style="background-color:green;">Serving with 2 x 8 x H200</mark>

1. Install SGLang following [the instruction](https://app.gitbook.com/s/FFtIWT8LEMaYiYzz0p8P/sglang-cookbook/installation/nvidia-h-series-a-series-and-rtx-gpus) for the 2 nodes
2. Serve the model

If the first node's IP is `10.0.0.1` , launch the server in both node with below commands

{% code overflow="wrap" %}
```bash
# node 1
python3 -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V3 --tp 16 --dist-init-addr 10.0.0.1:5000 --nnodes 2 --node-rank 0 --trust-remote-code --host 0.0.0.0 --port 30000

# node 2
python3 -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V3 --tp 16 --dist-init-addr 10.0.0.1:5000 --nnodes 2 --node-rank 1 --trust-remote-code --host 0.0.0.0 --port 30000
```
{% endcode %}

* [Optional Optimization Options](./#optional-performance-optimization)

### <mark style="background-color:green;">Serving with 4 x 8 x A100</mark>

1. Install SGLang following [the instruction](https://app.gitbook.com/s/FFtIWT8LEMaYiYzz0p8P/sglang-cookbook/installation/nvidia-h-series-a-series-and-rtx-gpus) for the 4 nodes
2. As A100 does not support FP8, we need to convert the [FP8 model checkpoints](https://huggingface.co/deepseek-ai/DeepSeek-V3) to BF16 with [script](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/fp8_cast_bf16.py) mentioned [here](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/fp8_cast_bf16.py) first
3. Serve the model

If the first node's IP is `10.0.0.1` , and the converted model path is `/path/to/DeepSeek-V3-BF16`, launch the server in 4 nodes with below commands

{% code overflow="wrap" %}
```bash
# node 1
python3 -m sglang.launch_server --model-path /path/to/DeepSeek-V3-BF16 --tp 32 --dist-init-addr 10.0.0.1:5000 --nnodes 4 --node-rank 0 --trust-remote-code --host 0.0.0.0 --port 30000

# node 2
python3 -m sglang.launch_server --model-path /path/to/DeepSeek-V3-BF16 --tp 32 --dist-init-addr 10.0.0.1:5000 --nnodes 4 --node-rank 1 --trust-remote-code

# node 3
python3 -m sglang.launch_server --model-path /path/to/DeepSeek-V3-BF16 --tp 32 --dist-init-addr 10.0.0.1:5000 --nnodes 4 --node-rank 2 --trust-remote-code

# node 4
python3 -m sglang.launch_server --model-path /path/to/DeepSeek-V3-BF16 --tp 32 --dist-init-addr 10.0.0.1:5000 --nnodes 4 --node-rank 3 --trust-remote-code
```
{% endcode %}

* [Optional Optimization Options](./#optional-performance-optimization)

### <mark style="background-color:green;">Serving with 8 x A100</mark>

1. Install SGLang following [the instruction](https://app.gitbook.com/s/FFtIWT8LEMaYiYzz0p8P/sglang-cookbook/installation/nvidia-h-series-a-series-and-rtx-gpus)
2. Serve the model

{% code overflow="wrap" %}
```bash
python3 -m sglang.launch_server --model cognitivecomputations/DeepSeek-R1-AWQ --tp 8 --trust-remote-code --quantization moe_wna16
```
{% endcode %}

Add `--quantization moe_wna16` flag to enable moe wna16 kernel for better performance. One example is as follows:

Alternatively, you can use `--quantization awq_marlin` as follows:

{% code overflow="wrap" %}
```bash
python3 -m sglang.launch_server --model cognitivecomputations/DeepSeek-R1-AWQ --tp 8 --trust-remote-code --quantization awq_marlin --dtype float16
```
{% endcode %}

Note that `awq_marlin` only supports `float16` now, which may lead to some precision loss.

### <mark style="background-color:green;">Serving with 2 x 8 x A100/A800</mark>

1. Install SGLang following [the instruction](https://app.gitbook.com/s/FFtIWT8LEMaYiYzz0p8P/sglang-cookbook/installation/nvidia-h-series-a-series-and-rtx-gpus) for the 4 nodes
2. Serve the model

There are block-wise and per-channel quantization methods, weights have already been quantized in these huggingface checkpoint:

* [meituan/DeepSeek-R1-Block-INT8](https://huggingface.co/meituan/DeepSeek-R1-Block-INT8)
* meituan/DeepSeek-R1-Channel-INT8

If the first node's IP is `10.0.0.1` , launch the server with below commands

{% code overflow="wrap" %}
```bash
#node 1
python3 -m sglang.launch_server \
	--model meituan/DeepSeek-R1-Block-INT8 --tp 16 --dist-init-addr \
	10.0.0.1:5000 --nnodes 2 --node-rank 0 --trust-remote-code --enable-torch-compile --torch-compile-max-bs 8

#node 2
python3 -m sglang.launch_server \
	--model meituan/DeepSeek-R1-Block-INT8 --tp 16 --dist-init-addr \
	10.0.0.1:5000 --nnodes 2 --node-rank 1 --trust-remote-code --enable-torch-compile --torch-compile-max-bs 8
```
{% endcode %}

* [Optional Optimization Options](./#optional-performance-optimization)

### <mark style="background-color:green;">Serving with 4 x 8 x L40S nodes</mark>

1. Install SGLang following [the instruction](https://app.gitbook.com/s/FFtIWT8LEMaYiYzz0p8P/sglang-cookbook/installation/nvidia-h-series-a-series-and-rtx-gpus) for the 4 nodes
2. Serve the model

Running with per-channel quantization model:

* [meituan/DeepSeek-R1-Channel-INT8](https://huggingface.co/meituan/DeepSeek-R1-Channel-INT8)

If the first node's IP is `10.0.0.1` , launch the server with below commands

{% code overflow="wrap" %}
```bash
#node 1
python3 -m sglang.launch_server --model meituan/DeepSeek-R1-Channel-INT8 --tp 32 --quantization w8a8_int8 \
	--dist-init-addr 10.0.0.1:5000 --nnodes 4 --node-rank 0 --trust-remote \
	--enable-torch-compile --torch-compile-max-bs 32
#node 2
python3 -m sglang.launch_server --model meituan/DeepSeek-R1-Channel-INT8 --tp 32 --quantization w8a8_int8 \
	--dist-init-addr 10.0.0.1:5000 --nnodes 4 --node-rank 1 --trust-remote \
	--enable-torch-compile --torch-compile-max-bs 32
#node 3
python3 -m sglang.launch_server --model meituan/DeepSeek-R1-Channel-INT8 --tp 32 --quantization w8a8_int8 \
	--dist-init-addr 10.0.0.1:5000 --nnodes 4 --node-rank 2 --trust-remote \
	--enable-torch-compile --torch-compile-max-bs 32
#node 4
python3 -m sglang.launch_server --model meituan/DeepSeek-R1-Channel-INT8 --tp 32 --quantization w8a8_int8 \
	--dist-init-addr 10.0.0.1:5000 --nnodes 4 --node-rank 3 --trust-remote \
	--enable-torch-compile --torch-compile-max-bs 32
```
{% endcode %}

### <mark style="background-color:green;">Example: Serving on any cloud or Kubernetes with SkyPilot</mark>

SkyPilot helps find cheapest available GPUs across any cloud or existing Kubernetes clusters and launch distributed serving with a single command. See details [here](https://github.com/skypilot-org/skypilot/tree/master/llm/deepseek-r1).

To serve on multiple nodes:

{% code overflow="wrap" %}
```bash
git clone https://github.com/skypilot-org/skypilot.git
# Serve on 2 H100/H200x8 nodes
sky launch -c r1 llm/deepseek-r1/deepseek-r1-671B.yaml --retry-until-up
# Serve on 4 A100x8 nodes
sky launch -c r1 llm/deepseek-r1/deepseek-r1-671B-A100.yaml --retry-until-up
```
{% endcode %}
