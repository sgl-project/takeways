# DeepSeek V3.1/V3/R1

<table><thead><tr><th>Weight Type</th><th width="249">Hardware Configuration</th><th data-type="content-ref">Instruction</th></tr></thead><tbody><tr><td><strong>Full precision FP8</strong><br><em>(recommended)</em></td><td>8 x H200</td><td><a href="usage-guide.md#serving-with-1-x-8-x-h200">#serving-with-1-x-8-x-h200</a></td></tr><tr><td></td><td>8 x MI300X</td><td><a href="usage-guide.md#serving-with-1-x-8-x-mi300x">#serving-with-1-x-8-x-mi300x</a></td></tr><tr><td></td><td>2 x 8 x H100/800/20</td><td><a href="usage-guide.md#serving-with-2-x-8-x-h100-800-20">#serving-with-2-x-8-x-h100-800-20</a></td></tr><tr><td></td><td>Xeon 6980P CPU</td><td><a href="usage-guide.md#serving-with-xeon-6980p-cpu">#serving-with-xeon-6980p-cpu</a></td></tr><tr><td><strong>Full precision BF16</strong></td><td>2 x 8 x H200</td><td><a href="usage-guide.md#serving-with-2-x-8-x-h200">#serving-with-2-x-8-x-h200</a></td></tr><tr><td></td><td>2 x 8 x MI300X</td><td></td></tr><tr><td></td><td>4 x 8 x H100/800/20</td><td></td></tr><tr><td></td><td>4 x 8 x A100/A800</td><td><a href="usage-guide.md#serving-with-4-x-8-x-a100">#serving-with-4-x-8-x-a100</a></td></tr><tr><td><strong>Quantized weights (AWQ)</strong></td><td>8 x H100/800/20</td><td></td></tr><tr><td></td><td>8 x A100/A800</td><td><a href="usage-guide.md#serving-with-8-x-a100">#serving-with-8-x-a100</a></td></tr><tr><td><strong>Quantized weights (int8)</strong></td><td>2 x 8 x A100/800</td><td><a href="usage-guide.md#serving-with-2-x-8-x-a100-a800">#serving-with-2-x-8-x-a100-a800</a></td></tr><tr><td></td><td>4 x 8 x L40S</td><td><a href="usage-guide.md#serving-with-4-x-8-x-l40s-nodes">#serving-with-4-x-8-x-l40s-nodes</a></td></tr><tr><td></td><td>Xeon 6980P CPU</td><td><a href="usage-guide.md#serving-with-xeon-6980p-cpu">#serving-with-xeon-6980p-cpu</a></td></tr><tr><td></td><td>2 x Atlas 800I A3</td><td></td></tr></tbody></table>

#### &#x20;Optional performance optimization

* [MLA optimizations](https://lmsys.org/blog/2024-09-04-sglang-v0-3/#deepseek-multi-head-latent-attention-mla-throughput-optimizations) are enabled by default
* [Data Parallelism Attention](https://lmsys.org/blog/2024-12-04-sglang-v0-4/#data-parallelism-attention-for-deepseek-models): For high QPS scenarios, add the `--enable-dp-attention` argument to boost throughput.
* [Torch.compile Optimization](https://lmsys.org/blog/2024-09-04-sglang-v0-3/#torchcompile-latency-optimizations): Add `--enable-torch-compile` argument to enable it. This will take some time while server starts. The maximum batch size for torch.compile optimization can be controlled with `--torch-compile-max-bs`. It's recommended to set it between `1` and `8`. (e.g., `--torch-compile-max-bs 8`)

#### **Troubleshooting**

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

#### DeepSeek V3 Optimization Plan

https://github.com/sgl-project/sglang/issues/2591
