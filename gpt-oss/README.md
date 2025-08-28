## `gpt-oss` SGLang Usage Guide

`gpt-oss-20b` and `gpt-oss-120b` are open-sourced models by OpenAI.

## Quick Start

### Prepare Docker

- H100/H200:
    
    ```bash
    # Start docker
    docker run -it --gpus all \
      --shm-size 32g \
      --network=host \
      -v ~/.cache/huggingface:/root/.cache/huggingface \
      --env "HF_TOKEN=<secret>" \
      --privileged \
      --ipc=host \
      --name sglang \
      lmsysorg/sglang:latest \
      /bin/zsh
      
      # in terminal, run
      python3 -m sglang.launch_server --model-path openai/gpt-oss-120b --tp 4
    ```
    
- B200:
    
    ```bash
    # Start docker
    docker run -it --gpus all \
      --shm-size 32g \
      --network=host \
      -v ~/.cache/huggingface:/root/.cache/huggingface \
      --env "HF_TOKEN=<secret>" \
      --privileged \
      --ipc=host \
      --name sglang \
      lmsysorg/sglang:blackwell \
      /bin/zsh
      
      # in terminal, run
      python3 -m sglang.launch_server --model-path openai/gpt-oss-120b --tp 4
    ```
    

### Launch Server

- Without speculative decoding
    
    ```bash
    # MXFP4 120B
    python3 -m sglang.launch_server --model openai/gpt-oss-120b --tp 4
    # BF16 120B
    python3 -m sglang.launch_server --model lmsys/gpt-oss-120b-bf16 --tp 4
    ```
    
- With speculative decoding
    
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
    

## Responses API & Built-in Tools

Please refer to https://docs.sglang.ai/basic_usage/gpt_oss.html#responses-api-built-in-tools

## Benchmark & Accuracy Test

### Benchmark Command

```bash
pip install tabulate

# Batch size 1
python3 -m sglang.bench_one_batch_server --model openai/gpt-oss-120b --base-url http://localhost:30000 --batch-size 1 --input-len 1024 --output-len 512

# Batch size 32
python3 -m sglang.bench_one_batch_server --model openai/gpt-oss-120b --base-url http://localhost:30000 --batch-size 32 --input-len 1024 8192 --output-len 512 --show-report
```

### Test Accuracy

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

## Known Limitations

- On Blackwell, the default attention backend (`trtllm_mha`) does not support `topk > 1` when speculative decoding is enabled. To use `topk > 1`, you should switch to a different attention backend (e.g., Triton) by specifying `--attention-backend triton`