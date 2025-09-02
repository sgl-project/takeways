# NVIDIA Blackwell GPUs

The docker images are available on Docker Hub at [lmsysorg/sglang](https://hub.docker.com/r/lmsysorg/sglang/tags). Replace `<secret>` below with your huggingface hub [token](https://huggingface.co/docs/hub/en/security-tokens).

{% code overflow="wrap" %}
```bash
docker run -it --gpus all \
  --shm-size 32g \
  --network=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --env "HF_TOKEN=<secret>" \
  --privileged \
  --ipc=host \
  --name sglang \
  lmsysorg/sglang:blackwell \
  /bin/bash
```
{% endcode %}

## B200 with x86 CPUs

TODO

## GB200/GB300 with ARM CPUs

TODO
