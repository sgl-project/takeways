---
icon: hat-chef
layout:
  width: default
  title:
    visible: true
  description:
    visible: true
  tableOfContents:
    visible: false
  outline:
    visible: false
  pagination:
    visible: true
  metadata:
    visible: false
---

# SGLang Cookbook

SGLang offers a wide array of configuration parameters. [This manual](https://docs.sglang.ai/) provides detailed coverage, but what users often need is a straightforward, ready-to-use setup that delivers strong performance, particularly when low latency is essential. The cookbook aim to answer the question:&#x20;

> <mark style="background-color:green;">**How do i serve model X in hardware Y?**</mark>

## Step 1 Install SGLang on the <mark style="background-color:green;">**hardware Y**</mark>

<table data-card-size="large" data-view="cards" data-full-width="false"><thead><tr><th></th><th data-hidden data-card-cover data-type="image">Cover image</th><th data-hidden data-card-target data-type="content-ref"></th></tr></thead><tbody><tr><td>NVIDIA H-Series, A-Series, and RTX GPUs</td><td><a href="../.gitbook/assets/01-nvidia-logo-horiz-500x200-2c50-d.png">01-nvidia-logo-horiz-500x200-2c50-d.png</a></td><td><a href="installation/nvidia-h-series-a-series-and-rtx-gpus.md">nvidia-h-series-a-series-and-rtx-gpus.md</a></td></tr><tr><td>NVIDIA Blackwell/AMD/Intel/NVIDIA Jetson/Ascend hardware</td><td><a href="../.gitbook/assets/download.jpeg">download.jpeg</a></td><td><a href="installation/">installation</a></td></tr></tbody></table>

SGLang's official model releases typically focus on H100 and B200 GPUs. However, the completeness of this cookbook relies on the open source community. We welcome contributions that extend support beyond these hardware specifications. Contributors can follow the format for existing modes, please also attach benchmarks results using our unified benchmark framework [genai-bench](https://github.com/sgl-project/genai-bench)➚

## Step 2 Serve the <mark style="background-color:green;">model X</mark>

<table data-view="cards"><thead><tr><th>Model</th><th data-hidden data-card-target data-type="content-ref"></th><th data-hidden data-card-cover data-type="image">Cover image</th></tr></thead><tbody><tr><td>DeepSeek V3.1/V3/R1</td><td><a href="deepseek-v3.1-v3-r1/">deepseek-v3.1-v3-r1</a></td><td><a href="../.gitbook/assets/deepseek-logo-icon.png">deepseek-logo-icon.png</a></td></tr><tr><td>Meta Llama</td><td><a href="meta-llama.md">meta-llama.md</a></td><td><a href="../.gitbook/assets/Llama-2-Model-Details.webp">Llama-2-Model-Details.webp</a></td></tr><tr><td>GPT-OSS</td><td><a href="gpt-oss/">gpt-oss</a></td><td><a href="../.gitbook/assets/20250102113948245.png">20250102113948245.png</a></td></tr></tbody></table>

## Step 3 Use the model

{% content-ref url="use-the-model/" %}
[use-the-model](use-the-model/)
{% endcontent-ref %}

