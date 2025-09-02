# NVIDIA H-Series, A-Series, and RTX GPUs

## Prerequisite

* OS: Linux
* Python: 3.9 -- 3.13

## Method 1: Using docker

The docker images are available on Docker Hub at [lmsysorg/sglang](https://hub.docker.com/r/lmsysorg/sglang/tags), built from [Dockerfile](https://github.com/sgl-project/sglang/tree/main/docker). Replace `<secret>` below with your huggingface hub [token](https://huggingface.co/docs/hub/en/security-tokens).

{% code overflow="wrap" %}
```bash
# Pull latest image
# https://hub.docker.com/r/lmsysorg/sglang/tags
docker pull lmsysorg/sglang:latest

docker run --gpus all \
    --shm-size 32g \
    -p 30000:30000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=<secret>" \
    --ipc=host \
    lmsysorg/sglang:latest \
    /bin/bash
```
{% endcode %}

## Method 2: With pip or uv

It is recommended to use uv for faster installation:

{% code overflow="wrap" %}
```bash
pip install --upgrade pip
pip install uv
uv pip install "sglang[all]>=0.5.1.post3"
```
{% endcode %}

## Method 3: From source

{% code overflow="wrap" %}
```bash
# Use the last release branch
git clone -b v0.5.1.post3 https://github.com/sgl-project/sglang.git
cd sglang

# Install the python packages
pip install --upgrade pip
pip install -e "python[all]"
```
{% endcode %}

## **Quick fixes to common problems**

* If you encounter `OSError: CUDA_HOME environment variable is not set`. Please set it to your CUDA install root with either of the following solutions:
  1. Use `export CUDA_HOME=/usr/local/cuda-<your-cuda-version>` to set the `CUDA_HOME` environment variable.
  2. Install FlashInfer first following [FlashInfer installation doc](https://docs.flashinfer.ai/installation.html), then install SGLang as described above.

## Notes

* If you only need to use OpenAI API models with the frontend language, you can avoid installing other dependencies by using `pip install "sglang[openai]"`.
* [FlashInfer](https://github.com/flashinfer-ai/flashinfer) is the default attention kernel backend. It only supports sm75 and above. If you encounter any FlashInfer-related issues on sm75+ devices (e.g., T4, A10, A100, L4, L40S, H100), please switch to other kernels by adding `--attention-backend triton --sampling-backend pytorch` and open an issue on GitHub.
  * To reinstall flashinfer locally, use the following command: `pip3 install --upgrade flashinfer-python --force-reinstall --no-deps` and then delete the cache with `rm -rf ~/.cache/flashinfer`.
* The language frontend operates independently of the backend runtime. You can install the frontend locally without needing a GPU, while the backend can be set up on a GPU-enabled machine. To install the frontend, run `pip install sglang`, and for the backend, use `pip install sglang[srt]`. `srt` is the abbreviation of SGLang runtime.
