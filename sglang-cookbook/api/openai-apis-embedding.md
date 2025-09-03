# OpenAI APIs - Embedding

SGLang provides OpenAI-compatible APIs to enable a smooth transition from OpenAI services to self-hosted local models. A complete reference for the API is available in the [OpenAI API Reference](https://platform.openai.com/docs/guides/embeddings).

This tutorial covers the embedding APIs for embedding models. For a list of the supported models see the [corresponding overview page](https://docs.sglang.ai/supported_models/embedding_models.html)

## Using OpenAI Python Client

```python
import openai

client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="None")

# Text embedding example
response = client.embeddings.create(
    model="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
    input=text,
)

embedding = response.data[0].embedding[:10]
print_highlight(f"Text embedding (first 10): {embedding}")
```

## Using Input IDs

SGLang also supports `input_ids` as input to get the embedding.

```python
import json
import os
from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

tokenizer = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-Qwen2-1.5B-instruct")
input_ids = tokenizer.encode(text)

curl_ids = f"""curl -s http://localhost:{port}/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{{"model": "Alibaba-NLP/gte-Qwen2-1.5B-instruct", "input": {json.dumps(input_ids)}}}'"""

input_ids_embedding = json.loads(subprocess.check_output(curl_ids, shell=True))["data"][
    0
]["embedding"]

print_highlight(f"Input IDs embedding (first 10): {input_ids_embedding[:10]}")
```

## Multi-Modal Embedding Model

Please refer to [Multi-Modal Embedding Model](https://docs.sglang.ai/supported_models/embedding_models.html)
