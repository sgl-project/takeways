# Usage Guide

### <mark style="background-color:green;">Serving with 1 x  8 x H200</mark>

#### Using Docker (Recommended)

{% code overflow="wrap" %}
```bash
# Pull latest image
# https://hub.docker.com/r/lmsysorg/sglang/tags
docker pull lmsysorg/sglang:latest

# Launch
docker run --gpus all --shm-size 32g -p 30000:30000 -v ~/.cache/huggingface:/root/.cache/huggingface --ipc=host --network=host --privileged lmsysorg/sglang:latest \
    python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-V3 --tp 8 --trust-remote-code --port 30000
```
{% endcode %}

If you are using RDMA, please note that:

1. `--network host` and `--privileged` are required by RDMA. If you don't need RDMA, you can remove them.
2. You may need to set `NCCL_IB_GID_INDEX` if you are using RoCE, for example: `export NCCL_IB_GID_INDEX=3`.

Add performance optimization options as needed.

#### Using pip

{% code overflow="wrap" %}
```bash
# Installation
pip install "sglang[all]>=0.5.1.post2"

# Launch
python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-V3 --tp 8 --trust-remote-code
```
{% endcode %}

#### Add performance optimization options as needed.

[MLA optimizations](https://lmsys.org/blog/2024-09-04-sglang-v0-3/#deepseek-multi-head-latent-attention-mla-throughput-optimizations) are enabled by default. Here are some optional optimizations can be enabled as needed.

* [Data Parallelism Attention](https://lmsys.org/blog/2024-12-04-sglang-v0-4/#data-parallelism-attention-for-deepseek-models): For high QPS scenarios, add the `--enable-dp-attention` argument to boost throughput.
* [Torch.compile Optimization](https://lmsys.org/blog/2024-09-04-sglang-v0-3/#torchcompile-latency-optimizations): Add `--enable-torch-compile` argument to enable it. This will take some time while server starts. The maximum batch size for torch.compile optimization can be controlled with `--torch-compile-max-bs`. It's recommended to set it between `1` and `8`. (e.g., `--torch-compile-max-bs 8`)

### <mark style="background-color:green;">Serving with 1 x  8 x MI300X</mark>

#### Using Docker (Recommended)

{% code overflow="wrap" %}
```bash
docker build -t sglang_image -f Dockerfile.rocm .
alias drun='docker run -it --rm --network=host --privileged --device=/dev/kfd --device=/dev/dri \
    --ipc=host --shm-size 16G --group-add video --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -v $HOME/dockerx:/dockerx \
    -v /data:/data'
drun -p 30000:30000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    --env "HF_TOKEN=<secret>" \
    sglang_image \
    python3 -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-V3 \ # <- here
    --tp 8 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 30000
```
{% endcode %}

If you are using RDMA, please note that:

1. `--network host` and `--privileged` are required by RDMA. If you don't need RDMA, you can remove them.
2. You may need to set `NCCL_IB_GID_INDEX` if you are using RoCE, for example: `export NCCL_IB_GID_INDEX=3`.

### <mark style="background-color:green;">Serving with 2 x 8 x H20</mark>

For example, there are two H20 nodes, each with 8 GPUs. The first node's IP is `10.0.0.1`, and the second node's IP is `10.0.0.2`. Please **use the first node's IP** for both commands.

If the command fails, try setting the `GLOO_SOCKET_IFNAME` parameter. For more information, see [Common Environment Variables](https://pytorch.org/docs/stable/distributed.html#common-environment-variables).

If the multi nodes support NVIDIA InfiniBand and encounter hanging issues during startup, consider adding the parameter `export NCCL_IB_GID_INDEX=3`. For more information, see [this](https://github.com/sgl-project/sglang/issues/3516#issuecomment-2668493307).

{% code overflow="wrap" %}
```bash
# node 1
python3 -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V3 --tp 16 --dist-init-addr 10.0.0.1:5000 --nnodes 2 --node-rank 0 --trust-remote-code

# node 2
python3 -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V3 --tp 16 --dist-init-addr 10.0.0.1:5000 --nnodes 2 --node-rank 1 --trust-remote-code
```
{% endcode %}

If you have two H100 nodes, the usage is similar to the aforementioned H20.

> **Note that the launch command here does not enable Data Parallelism Attention or `torch.compile` Optimization**. For optimal performance, please refer to the command options in Performance Optimization Options.

### <mark style="background-color:green;">Serving with 2 x 8 x H200</mark>

There are two H200 nodes, each with 8 GPUs. The first node's IP is `192.168.114.10`, and the second node's IP is `192.168.114.11`. Configure the endpoint to expose it to another Docker container using `--host 0.0.0.0` and `--port 40000`, and set up communications with `--dist-init-addr 192.168.114.10:20000`.&#x20;

{% code overflow="wrap" %}
```bash
# node 1
docker run --gpus all \
    --shm-size 32g \
    --network=host \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --name sglang_multinode1 \
    -it \
    --rm \
    --env "HF_TOKEN=$HF_TOKEN" \
    --ipc=host \
    lmsysorg/sglang:latest \
    python3 -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V3 --tp 16 --dist-init-addr 192.168.114.10:20000 --nnodes 2 --node-rank 0 --trust-remote-code --host 0.0.0.0 --port 40000
```
{% endcode %}

{% code overflow="wrap" %}
```bash
# node 2
docker run --gpus all \
    --shm-size 32g \
    --network=host \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --name sglang_multinode2 \
    -it \
    --rm \
    --env "HF_TOKEN=$HF_TOKEN" \
    --ipc=host \
    lmsysorg/sglang:latest \
    python3 -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V3 --tp 16 --dist-init-addr 192.168.114.10:20000 --nnodes 2 --node-rank 1 --trust-remote-code --host 0.0.0.0 --port 40000
```
{% endcode %}

To ensure functionality, we include a test from a client Docker container.

{% code overflow="wrap" %}
```bash
docker run --gpus all \
    --shm-size 32g \
    --network=host \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --name sglang_multinode_client \
    -it \
    --rm \
    --env "HF_TOKEN=$HF_TOKEN" \
    --ipc=host \
    lmsysorg/sglang:latest \
    python3 -m sglang.bench_serving --backend sglang --dataset-name random --random-input 1 --random-output 512 --random-range-ratio 1 --num-prompts 1 --host 0.0.0.0 --port 40000 --output-file "deepseekv3_multinode.jsonl"
```
{% endcode %}

> **Note that the launch command here does not enable Data Parallelism Attention or `torch.compile` Optimization**. For optimal performance, please refer to the command options in Performance Optimization Options.

### <mark style="background-color:green;">Serving with 4 x 8 x A100</mark>

To serve DeepSeek-V3 with A100 GPUs, we need to convert the [FP8 model checkpoints](https://huggingface.co/deepseek-ai/DeepSeek-V3) to BF16 with [script](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/fp8_cast_bf16.py) mentioned [here](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/fp8_cast_bf16.py) first.

Since the BF16 model is over 1.3 TB, we need to prepare four A100 nodes, each with 8 80GB GPUs. Assume the first node's IP is `10.0.0.1`, and the converted model path is `/path/to/DeepSeek-V3-BF16`, we can have following commands to launch the server.

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

> **Note that the launch command here does not enable Data Parallelism Attention or `torch.compile` Optimization**. For optimal performance, please refer to the command options in Performance Optimization Options.

Then we can benchmark the accuracy and latency by accessing the first node's exposed port with the following example commands.

{% code overflow="wrap" %}
```bash
# bench accuracy
python3 benchmark/gsm8k/bench_sglang.py --num-questions 1319 --host http://10.0.0.1 --port 30000

# bench latency
python3 -m sglang.bench_one_batch_server --model None --base-url http://10.0.0.1:30000 --batch-size 1 --input-len 128 --output-len 128
```
{% endcode %}

### <mark style="background-color:green;">Serving with 8 x A100</mark>

**Recommended Usage**

Add `--quantization moe_wna16` flag to enable moe wna16 kernel for better performance. One example is as follows:

{% code overflow="wrap" %}
```bash
python3 -m sglang.launch_server --model cognitivecomputations/DeepSeek-R1-AWQ --tp 8 --trust-remote-code --quantization moe_wna16
```
{% endcode %}

Alternatively, you can use `--quantization awq_marlin` as follows:

{% code overflow="wrap" %}
```bash
python3 -m sglang.launch_server --model cognitivecomputations/DeepSeek-R1-AWQ --tp 8 --trust-remote-code --quantization awq_marlin --dtype float16
```
{% endcode %}

Note that `awq_marlin` only supports `float16` now, which may lead to some precision loss.

### <mark style="background-color:green;">Serving with 16 x A100/A800</mark>

There are block-wise and per-channel quantization methods, and the quantization parameters have already been uploaded to Huggingface. One example is as follows:

* [meituan/DeepSeek-R1-Block-INT8](https://huggingface.co/meituan/DeepSeek-R1-Block-INT8)
* [meituan/DeepSeek-R1-Channel-INT8](https://huggingface.co/meituan/DeepSeek-R1-Channel-INT8)

Assuming that master node IP is `MASTER_IP`, checkpoint path is `/path/to/DeepSeek-R1-INT8` and port=5000, we can have following commands to launch the server:

{% code overflow="wrap" %}
```bash
#master
python3 -m sglang.launch_server \
	--model meituan/DeepSeek-R1-Block-INT8 --tp 16 --dist-init-addr \
	MASTER_IP:5000 --nnodes 2 --node-rank 0 --trust-remote-code --enable-torch-compile --torch-compile-max-bs 8
#cluster
python3 -m sglang.launch_server \
	--model meituan/DeepSeek-R1-Block-INT8 --tp 16 --dist-init-addr \
	MASTER_IP:5000 --nnodes 2 --node-rank 1 --trust-remote-code --enable-torch-compile --torch-compile-max-bs 8
```
{% endcode %}

> **Note that the launch command here enables `torch.compile` Optimization**. For optimal performance, please refer to the command options in Performance Optimization Options.

Then on the **master node**, supposing the ShareGPT data is located at `/path/to/ShareGPT_V3_unfiltered_cleaned_split.json`, you can run the following commands to benchmark the launched server:

{% code overflow="wrap" %}
```bash
# bench accuracy
python3 benchmark/gsm8k/bench_sglang.py --num-questions 1319

# bench serving
python3 -m sglang.bench_serving --dataset-path /path/to/ShareGPT_V3_unfiltered_cleaned_split.json --dataset-name random  --random-input 128 --random-output 128 --num-prompts 1000 --request-rate 128 --random-range-ratio 1.0
```
{% endcode %}

> **Note: using `--parallel 200` can accelerate accuracy benchmarking**.

### <mark style="background-color:green;">Serving with 32 x L40S nodes</mark>

Running with per-channel quantization model:

* [meituan/DeepSeek-R1-Channel-INT8](https://huggingface.co/meituan/DeepSeek-R1-Channel-INT8)

Assuming that master node IP is `MASTER_IP`, checkpoint path is `/path/to/DeepSeek-R1-Channel-INT8` and port=5000, we can have following commands to launch the server:

{% code overflow="wrap" %}
```bash
#master
python3 -m sglang.launch_server --model meituan/DeepSeek-R1-Channel-INT8 --tp 32 --quantization w8a8_int8 \
	--dist-init-addr MASTER_IP:5000 --nnodes 4 --node-rank 0 --trust-remote \
	--enable-torch-compile --torch-compile-max-bs 32
#cluster
python3 -m sglang.launch_server --model meituan/DeepSeek-R1-Channel-INT8 --tp 32 --quantization w8a8_int8 \
	--dist-init-addr MASTER_IP:5000 --nnodes 4 --node-rank 1 --trust-remote \
	--enable-torch-compile --torch-compile-max-bs 32
python3 -m sglang.launch_server --model meituan/DeepSeek-R1-Channel-INT8 --tp 32 --quantization w8a8_int8 \
	--dist-init-addr MASTER_IP:5000 --nnodes 4 --node-rank 2 --trust-remote \
	--enable-torch-compile --torch-compile-max-bs 32
python3 -m sglang.launch_server --model meituan/DeepSeek-R1-Channel-INT8 --tp 32 --quantization w8a8_int8 \
	--dist-init-addr MASTER_IP:5000 --nnodes 4 --node-rank 3 --trust-remote \
	--enable-torch-compile --torch-compile-max-bs 32
```
{% endcode %}

The benchmarking method is the same as describted in the previous [16 x A100](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#example-serving-with-16-a100a800-with-int8-quantization) example.

### <mark style="background-color:green;">Serving with Xeon 6980P CPU</mark>

#### Using Docker (Recommended)

A [Dockerfile](https://github.com/sgl-project/sglang/blob/main/docker/Dockerfile.xeon) is provided to facilitate the installation. Replace `<secret>` below with your [HuggingFace access token](https://huggingface.co/docs/hub/en/security-tokens).

{% code overflow="wrap" %}
```bash
# Clone the SGLang repository
git clone https://github.com/sgl-project/sglang.git
cd sglang/docker

# Build the docker image
docker build -t sglang-cpu:main -f Dockerfile.xeon .

# Initiate a docker container
docker run \
    -it \
    --privileged \
    --ipc=host \
    --network=host \
    -v /dev/shm:/dev/shm \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 30000:30000 \
    -e "HF_TOKEN=<secret>" \
    sglang-cpu:main /bin/bash
```
{% endcode %}

For W8A8

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

For FP8

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



#### Example: Serving on any cloud or Kubernetes with SkyPilot

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

**Troubleshooting**

If you encounter the following error with fp16/bf16 checkpoint:

{% code overflow="wrap" %}
```bash
ValueError: Weight output_partition_size = 576 is not divisible by weight quantization block_n = 128.
```
{% endcode %}

edit your `config.json` and remove the `quantization_config` block. For example:

```json
"quantization_config": {
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "quant_method": "fp8",
    "weight_block_size": [128, 128]
},
```

Removing this block typically resolves the error. For more details, see the discussion in [sgl-project/sglang#3491](https://github.com/sgl-project/sglang/issues/3491#issuecomment-2650779851).
