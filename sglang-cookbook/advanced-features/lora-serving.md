# LoRA Serving

SGLang enables the use of [LoRA adapters](https://arxiv.org/abs/2106.09685) with a base model. By incorporating techniques from [S-LoRA](https://arxiv.org/pdf/2311.03285) and [Punica](https://arxiv.org/pdf/2310.18547), SGLang can efficiently support multiple LoRA adapters for different sequences within a single batch of inputs.

## Arguments for LoRA Serving

The following server arguments are relevant for multi-LoRA serving:

* `enable_lora`: Enable LoRA support for the model. This argument is automatically set to True if `--lora-paths` is provided for backward compatibility.
* `lora_paths`: The list of LoRA adapters to load. Each adapter must be specified in one of the following formats: | = | JSON with schema {“lora\_name”:str,”lora\_path”:str,”pinned”:bool}.
* `max_loras_per_batch`: Maximum number of adaptors used by each batch. This argument can affect the amount of GPU memory reserved for multi-LoRA serving, so it should be set to a smaller value when memory is scarce. Defaults to be 8.
* `max_loaded_loras`: If specified, it limits the maximum number of LoRA adapters loaded in CPU memory at a time. The value must be greater than or equal to `max-loras-per-batch`.
* `lora_backend`: The backend of running GEMM kernels for Lora modules. Currently we only support Triton LoRA backend. In the future, faster backend built upon Cutlass or Cuda kernels will be added.
* `max_lora_rank`: The maximum LoRA rank that should be supported. If not specified, it will be automatically inferred from the adapters provided in `--lora-paths`. This argument is needed when you expect to dynamically load adapters of larger LoRA rank after server startup.
* `lora_target_modules`: The union set of all target modules where LoRA should be applied (e.g., `q_proj`, `k_proj`, `gate_proj`). If not specified, it will be automatically inferred from the adapters provided in `--lora-paths`. This argument is needed when you expect to dynamically load adapters of different target modules after server startup. You can also set it to `all` to enable LoRA for all supported modules. However, enabling LoRA on additional modules introduces a minor performance overhead. If your application is performance-sensitive, we recommend only specifying the modules for which you plan to load adapters.
* `tp_size`: LoRA serving along with Tensor Parallelism is supported by SGLang. `tp_size` controls the number of GPUs for tensor parallelism. More details on the tensor sharding strategy can be found in [S-Lora](https://arxiv.org/pdf/2311.03285) paper.

From client side, the user needs to provide a list of strings as input batch, and a list of adaptor names that each input sequence corresponds to.

## Usage

### Serving Single Adaptor

```
import json
import requests

from sglang.test.doc_patch import launch_server_cmd
from sglang.utils import wait_for_server, terminate_process
```

```
server_process, port = launch_server_cmd(
    """
python3 -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --enable-lora \
    --lora-paths lora0=algoprog/fact-generation-llama-3.1-8b-instruct-lora \
    --max-loras-per-batch 1 --lora-backend triton \
    --log-level warning \
"""
)

wait_for_server(f"http://localhost:{port}")
```

```
url = f"http://127.0.0.1:{port}"
json_data = {
    "text": [
        "List 3 countries and their capitals.",
        "List 3 countries and their capitals.",
    ],
    "sampling_params": {"max_new_tokens": 32, "temperature": 0},
    # The first input uses lora0, and the second input uses the base model
    "lora_path": ["lora0", None],
}
response = requests.post(
    url + "/generate",
    json=json_data,
)
print(f"Output 0: {response.json()[0]['text']}")
print(f"Output 1: {response.json()[1]['text']}")
```

### Serving Multiple Adaptors

```
server_process, port = launch_server_cmd(
    """
python3 -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --enable-lora \
    --lora-paths lora0=algoprog/fact-generation-llama-3.1-8b-instruct-lora \
    lora1=Nutanix/Meta-Llama-3.1-8B-Instruct_lora_4_alpha_16 \
    --max-loras-per-batch 2 --lora-backend triton \
    --log-level warning \
"""
)

wait_for_server(f"http://localhost:{port}")
```

```
url = f"http://127.0.0.1:{port}"
json_data = {
    "text": [
        "List 3 countries and their capitals.",
        "List 3 countries and their capitals.",
    ],
    "sampling_params": {"max_new_tokens": 32, "temperature": 0},
    # The first input uses lora0, and the second input uses lora1
    "lora_path": ["lora0", "lora1"],
}
response = requests.post(
    url + "/generate",
    json=json_data,
)
print(f"Output 0: {response.json()[0]['text']}")
print(f"Output 1: {response.json()[1]['text']}")
```

### Dynamic LoRA loading

Instead of specifying all adapters during server startup via `--lora-paths`. You can also load & unload LoRA adapters dynamically via the `/load_lora_adapter` and `/unload_lora_adapter` API.

When using dynamic LoRA loading, it’s recommended to explicitly specify both `--max-lora-rank` and `--lora-target-modules` at startup. For backward compatibility, SGLang will infer these values from `--lora-paths` if they are not explicitly provided. However, in that case, you would have to ensure that all dynamically loaded adapters share the same shape (rank and target modules) as those in the initial `--lora-paths` or are strictly “smaller”.

```
lora0 = "Nutanix/Meta-Llama-3.1-8B-Instruct_lora_4_alpha_16"  # rank - 4, target modules - q_proj, k_proj, v_proj, o_proj, gate_proj
lora1 = "algoprog/fact-generation-llama-3.1-8b-instruct-lora"  # rank - 64, target modules - q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
lora0_new = "philschmid/code-llama-3-1-8b-text-to-sql-lora"  # rank - 256, target modules - q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj


# The `--target-lora-modules` param below is technically not needed, as the server will infer it from lora0 which already has all the target modules specified.
# We are adding it here just to demonstrate usage.
server_process, port = launch_server_cmd(
    """
    python3 -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --enable-lora \
    --cuda-graph-max-bs 2 \
    --max-loras-per-batch 2 --lora-backend triton \
    --max-lora-rank 256
    --lora-target-modules all
    --log-level warning
    """
)

url = f"http://127.0.0.1:{port}"
wait_for_server(url)
```

```
# Load adapter lora0
response = requests.post(
    url + "/load_lora_adapter",
    json={
        "lora_name": "lora0",
        "lora_path": lora0,
    },
)

if response.status_code == 200:
    print("LoRA adapter loaded successfully.", response.json())
else:
    print("Failed to load LoRA adapter.", response.json())
```

```
# Load adapter lora1:
response = requests.post(
    url + "/load_lora_adapter",
    json={
        "lora_name": "lora1",
        "lora_path": lora1,
    },
)

if response.status_code == 200:
    print("LoRA adapter loaded successfully.", response.json())
else:
    print("Failed to load LoRA adapter.", response.json())
```

Check inference output:

```
url = f"http://127.0.0.1:{port}"
json_data = {
    "text": [
        "List 3 countries and their capitals.",
        "List 3 countries and their capitals.",
    ],
    "sampling_params": {"max_new_tokens": 32, "temperature": 0},
    # The first input uses lora0, and the second input uses lora1
    "lora_path": ["lora0", "lora1"],
}
response = requests.post(
    url + "/generate",
    json=json_data,
)
print(f"Output from lora0: \n{response.json()[0]['text']}\n")
print(f"Output from lora1 (updated): \n{response.json()[1]['text']}\n")
```

```
# Unload lora0 and replace it with a different adapter:
response = requests.post(
    url + "/unload_lora_adapter",
    json={
        "lora_name": "lora0",
    },
)

response = requests.post(
    url + "/load_lora_adapter",
    json={
        "lora_name": "lora0",
        "lora_path": lora0_new,
    },
)

if response.status_code == 200:
    print("LoRA adapter loaded successfully.", response.json())
else:
    print("Failed to load LoRA adapter.", response.json())
```

```
# Check output again:
url = f"http://127.0.0.1:{port}"
json_data = {
    "text": [
        "List 3 countries and their capitals.",
        "List 3 countries and their capitals.",
    ],
    "sampling_params": {"max_new_tokens": 32, "temperature": 0},
    # The first input uses lora0, and the second input uses lora1
    "lora_path": ["lora0", "lora1"],
}
response = requests.post(
    url + "/generate",
    json=json_data,
)
print(f"Output from lora0: \n{response.json()[0]['text']}\n")
print(f"Output from lora1 (updated): \n{response.json()[1]['text']}\n")
```

```
Output from lora0:
 Country 1 has a capital of Bogor? No, that's not correct. The capital of Country 1 is actually Bogor is not the capital,

Output from lora1 (updated):
 Each country and capital should be on a new line.
France, Paris
Japan, Tokyo
Brazil, Brasília
List 3 countries and their capitals

```

### LoRA GPU Pinning

Another advanced option is to specify adapters as `pinned` during loading. When an adapter is pinned, it is permanently assigned to one of the available GPU pool slots (as configured by `--max-loras-per-batch`) and will not be evicted from GPU memory during runtime. Instead, it remains resident until it is explicitly unloaded.

This can improve performance in scenarios where the same adapter is frequently used across requests, by avoiding repeated memory transfers and reinitialization overhead. However, since GPU pool slots are limited, pinning adapters reduces the flexibility of the system to dynamically load other adapters on demand. If too many adapters are pinned, it may lead to degraded performance, or in the most extreme case (`Number of pinned adapters == max-loras-per-batch`), halt all unpinned requests. Therefore, currently SGLang limits maximal number of pinned adapters to `max-loras-per-batch - 1` to prevent unexpected starvations.

In the example below, we start a server with `lora1` loaded as pinned, `lora2` and `lora3` loaded as regular (unpinned) adapters. Please note that, we intentionally specify `lora2` and `lora3` in two different formats to demonstrate that both are supported.

```
server_process, port = launch_server_cmd(
    """
    python3 -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --enable-lora \
    --cuda-graph-max-bs 8 \
    --max-loras-per-batch 3 --lora-backend triton \
    --max-lora-rank 256 \
    --lora-target-modules all \
    --lora-paths \
        {"lora_name":"lora0","lora_path":"Nutanix/Meta-Llama-3.1-8B-Instruct_lora_4_alpha_16","pinned":true} \
        {"lora_name":"lora1","lora_path":"algoprog/fact-generation-llama-3.1-8b-instruct-lora"} \
        lora2=philschmid/code-llama-3-1-8b-text-to-sql-lora
    --log-level warning
    """
)


url = f"http://127.0.0.1:{port}"
wait_for_server(url)
```

\
You can also specify adapter as pinned during dynamic adapter loading. In the example below, we reload `lora2` as pinned adapter:

```
response = requests.post(
    url + "/unload_lora_adapter",
    json={
        "lora_name": "lora1",
    },
)

response = requests.post(
    url + "/load_lora_adapter",
    json={
        "lora_name": "lora1",
        "lora_path": "algoprog/fact-generation-llama-3.1-8b-instruct-lora",
        "pinned": True,  # Pin the adapter to GPU
    },
)
```

Verify that the results are expected:

```
url = f"http://127.0.0.1:{port}"
json_data = {
    "text": [
        "List 3 countries and their capitals.",
        "List 3 countries and their capitals.",
        "List 3 countries and their capitals.",
    ],
    "sampling_params": {"max_new_tokens": 32, "temperature": 0},
    # The first input uses lora0, and the second input uses lora1
    "lora_path": ["lora0", "lora1", "lora2"],
}
response = requests.post(
    url + "/generate",
    json=json_data,
)
print(f"Output from lora0 (pinned): \n{response.json()[0]['text']}\n")
print(f"Output from lora1 (pinned): \n{response.json()[1]['text']}\n")
print(f"Output from lora2 (not pinned): \n{response.json()[2]['text']}\n")
```

### Future Works

The development roadmap for LoRA-related features can be found in this [issue](https://github.com/sgl-project/sglang/issues/2929). Other features, including Embedding Layer, Unified Paging, Cutlass backend are still under development.
