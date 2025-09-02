# Send your first request

This page is a quick-start guide to use SGLang in chat completions after installation.

* For Vision Language Models, see [OpenAI APIs - Vision](https://docs.sglang.ai/basic_usage/openai_api_vision.html).
* For Embedding Models, see [OpenAI APIs - Embedding](https://docs.sglang.ai/basic_usage/openai_api_embeddings.html) and [Encode (embedding model)](https://docs.sglang.ai/basic_usage/native_api.html#Encode-\(embedding-model\)).
* For Reward Models, see [Classify (reward model)](https://docs.sglang.ai/basic_usage/native_api.html#Classify-\(reward-model\)).

## Launch A Server

```python
from sglang.test.doc_patch import launch_server_cmd
from sglang.utils import wait_for_server, print_highlight, terminate_process

# This is equivalent to running the following command in your terminal
# python3 -m sglang.launch_server --model-path qwen/qwen2.5-0.5b-instruct --host 0.0.0.0

server_process, port = launch_server_cmd(
    """
python3 -m sglang.launch_server --model-path qwen/qwen2.5-0.5b-instruct \
 --host 0.0.0.0
"""
)

wait_for_server(f"http://localhost:{port}")
```

## Using cURL

```python
import subprocess, json

curl_command = f"""
curl -s http://localhost:{port}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{{"model": "qwen/qwen2.5-0.5b-instruct", "messages": [{{"role": "user", "content": "What is the capital of France?"}}]}}'
"""

response = json.loads(subprocess.check_output(curl_command, shell=True))
print_highlight(response)
```

## Using Python Requests

```python
import requests

url = f"http://localhost:{port}/v1/chat/completions"

data = {
    "model": "qwen/qwen2.5-0.5b-instruct",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
}

response = requests.post(url, json=data)
print_highlight(response.json())
```
