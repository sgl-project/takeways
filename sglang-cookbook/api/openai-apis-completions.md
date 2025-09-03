# OpenAI APIs - Completions

SGLang provides OpenAI-compatible APIs to enable a smooth transition from OpenAI services to self-hosted local models. A complete reference for the API is available in the [OpenAI API Reference](https://platform.openai.com/docs/api-reference).

This tutorial covers the following popular APIs:

* <mark style="color:$info;">`chat/completions`</mark>
* <mark style="color:$info;">`completions`</mark>

## Usage

The server fully implements the OpenAI API. It will automatically apply the chat template specified in the Hugging Face tokenizer, if one is available. You can also specify a custom chat template with <mark style="color:$info;">`--chat-template`</mark> when launching the server.

```python
import openai

client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="None")

response = client.chat.completions.create(
    model="qwen/qwen2.5-0.5b-instruct",
    messages=[
        {"role": "user", "content": "List 3 countries and their capitals."},
    ],
    temperature=0,
    max_tokens=64,
)

print_highlight(f"Response: {response}")
```

## Model Thinking/Reasoning Support

Some models support internal reasoning or thinking processes that can be exposed in the API response. SGLang provides unified support for various reasoning models through the `chat_template_kwargs` parameter and compatible reasoning parsers.

### **Supported Models and Configuration**

| Model Family                          | Chat Template Parameter                             | Reasoning Parser                                                      | Notes                                      |
| ------------------------------------- | --------------------------------------------------- | --------------------------------------------------------------------- | ------------------------------------------ |
| DeepSeek-R1 (R1, R1-0528, R1-Distill) | <mark style="color:$info;">`enable_thinking`</mark> | <mark style="color:$info;">`--reasoning-parser deepseek-r1`</mark>    | Standard reasoning models                  |
| DeepSeek-V3.1                         | <mark style="color:$info;">`thinking`</mark>        | <mark style="color:$info;">`--reasoning-parser deepseek-v3`</mark>    | Hybrid model (thinking/non-thinking modes) |
| Qwen3 (standard)                      | <mark style="color:$info;">`enable_thinking`</mark> | <mark style="color:$info;">`--reasoning-parser qwen3`</mark>          | Hybrid model (thinking/non-thinking modes) |
| Qwen3-Thinking                        | N/A (always enabled)                                | <mark style="color:$info;">`--reasoning-parser qwen3-thinking`</mark> | Always generates reasoning                 |
| Kimi                                  | N/A (always enabled)                                | <mark style="color:$info;">`--reasoning-parser kimi`</mark>           | Kimi thinking models                       |
| Gpt-Oss                               | N/A (always enabled)                                | <mark style="color:$info;">`--reasoning-parser gpt-oss`</mark>        | Gpt-Oss thinking models                    |

### **Basic Usage**

To enable reasoning output, you need to:

1. Launch the server with the appropriate reasoning parser
2. Set the model-specific parameter in <mark style="color:$info;">`chat_template_kwargs`</mark>
3. Optionally use <mark style="color:$info;">`separate_reasoning: False`</mark> to not get reasoning content separately (default to <mark style="color:$info;">`True`</mark>)

**Note for Qwen3-Thinking models:** These models always generate thinking content and do not support the <mark style="color:$info;">`enable_thinking`</mark> parameter. Use <mark style="color:$info;">`--reasoning-parser qwen3-thinking`</mark> or <mark style="color:$info;">`--reasoning-parser qwen3`</mark> to parse the thinking content.

#### **Example: Qwen3 Models**

```python
# Launch server:
# python3 -m sglang.launch_server --model Qwen/Qwen3-4B --reasoning-parser qwen3

from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url=f"http://127.0.0.1:30000/v1",
)

model = "Qwen/Qwen3-4B"
messages = [{"role": "user", "content": "How many r's are in 'strawberry'?"}]

response = client.chat.completions.create(
    model=model,
    messages=messages,
    extra_body={
        "chat_template_kwargs": {"enable_thinking": True},
        "separate_reasoning": True
    }
)

print("Reasoning:", response.choices[0].message.reasoning_content)
print("-"*100)
print("Answer:", response.choices[0].message.content)
```

**ExampleOutput:**

```
Reasoning: Okay, so the user is asking how many 'r's are in the word 'strawberry'. Let me think. First, I need to make sure I have the word spelled correctly. Strawberry... S-T-R-A-W-B-E-R-R-Y. Wait, is that right? Let me break it down.

Starting with 'strawberry', let's write out the letters one by one. S, T, R, A, W, B, E, R, R, Y. Hmm, wait, that's 10 letters. Let me check again. S (1), T (2), R (3), A (4), W (5), B (6), E (7), R (8), R (9), Y (10). So the letters are S-T-R-A-W-B-E-R-R-Y.
...
Therefore, the answer should be three R's in 'strawberry'. But I need to make sure I'm not counting any other letters as R. Let me check again. S, T, R, A, W, B, E, R, R, Y. No other R's. So three in total. Yeah, that seems right.

----------------------------------------------------------------------------------------------------
Answer: The word "strawberry" contains **three** letters 'r'. Here's the breakdown:

1. **S-T-R-A-W-B-E-R-R-Y**
   - The **third letter** is 'R'.
   - The **eighth and ninth letters** are also 'R's.

Thus, the total count is **3**.

**Answer:** 3.
```

**Note:** Setting <mark style="color:$info;">`"enable_thinking": False`</mark> (or omitting it) will result in <mark style="color:$info;">`reasoning_content`</mark> being <mark style="color:$info;">`None`</mark>. Qwen3-Thinking models always generate reasoning content and don’t support the <mark style="color:$info;">`enable_thinking`</mark> parameter.

#### **Example: DeepSeek-V3 Models**

DeepSeek-V3 models support thinking mode through the <mark style="color:$info;">`thinking`</mark> parameter:

```python
# Launch server:
# python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-V3.1 --tp 8  --reasoning-parser deepseek-v3

from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url=f"http://127.0.0.1:30000/v1",
)

model = "deepseek-ai/DeepSeek-V3.1"
messages = [{"role": "user", "content": "How many r's are in 'strawberry'?"}]

response = client.chat.completions.create(
    model=model,
    messages=messages,
    extra_body={
        "chat_template_kwargs": {"thinking": True},
        "separate_reasoning": True
    }
)

print("Reasoning:", response.choices[0].message.reasoning_content)
print("-"*100)
print("Answer:", response.choices[0].message.content)
```

**Example Output:**

```
Reasoning: First, the question is: "How many r's are in 'strawberry'?"

I need to count the number of times the letter 'r' appears in the word "strawberry".

Let me write out the word: S-T-R-A-W-B-E-R-R-Y.

Now, I'll go through each letter and count the 'r's.
...
So, I have three 'r's in "strawberry".

I should double-check. The word is spelled S-T-R-A-W-B-E-R-R-Y. The letters are at positions: 3, 8, and 9 are 'r's. Yes, that's correct.

Therefore, the answer should be 3.
----------------------------------------------------------------------------------------------------
Answer: The word "strawberry" contains **3** instances of the letter "r". Here's a breakdown for clarity:

- The word is spelled: S-T-R-A-W-B-E-R-R-Y
- The "r" appears at the 3rd, 8th, and 9th positions.
```

**Note:** DeepSeek-V3 models use the `thinking` parameter (not `enable_thinking`) to control reasoning output.

### Parameters

The chat completions API accepts OpenAI Chat Completions API’s parameters. Refer to [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat/create) for more details.

SGLang extends the standard API with the `extra_body` parameter, allowing for additional customization. One key option within `extra_body` is `chat_template_kwargs`, which can be used to pass arguments to the chat template processor.

```python
response = client.chat.completions.create(
    model="qwen/qwen2.5-0.5b-instruct",
    messages=[
        {
            "role": "system",
            "content": "You are a knowledgeable historian who provides concise responses.",
        },
        {"role": "user", "content": "Tell me about ancient Rome"},
        {
            "role": "assistant",
            "content": "Ancient Rome was a civilization centered in Italy.",
        },
        {"role": "user", "content": "What were their major achievements?"},
    ],
    temperature=0.3,  # Lower temperature for more focused responses
    max_tokens=128,  # Reasonable length for a concise response
    top_p=0.95,  # Slightly higher for better fluency
    presence_penalty=0.2,  # Mild penalty to avoid repetition
    frequency_penalty=0.2,  # Mild penalty for more natural language
    n=1,  # Single response is usually more stable
    seed=42,  # Keep for reproducibility
)

print_highlight(response.choices[0].message.content)
```

Streaming mode is also supported.

```python
stream = client.chat.completions.create(
    model="qwen/qwen2.5-0.5b-instruct",
    messages=[{"role": "user", "content": "Say this is a test"}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
```

## Completions

### Usage

Completions API is similar to Chat Completions API, but without the <mark style="color:$info;">`messages`</mark> parameter or chat templates.

```python
response = client.completions.create(
    model="qwen/qwen2.5-0.5b-instruct",
    prompt="List 3 countries and their capitals.",
    temperature=0,
    max_tokens=64,
    n=1,
    stop=None,
)

print_highlight(f"Response: {response}")
```

### Parameters

The completions API accepts OpenAI Completions API’s parameters. Refer to [OpenAI Completions API](https://platform.openai.com/docs/api-reference/completions/create) for more details.

Here is an example of a detailed completions request:

```python
response = client.completions.create(
    model="qwen/qwen2.5-0.5b-instruct",
    prompt="Write a short story about a space explorer.",
    temperature=0.7,  # Moderate temperature for creative writing
    max_tokens=150,  # Longer response for a story
    top_p=0.9,  # Balanced diversity in word choice
    stop=["\n\n", "THE END"],  # Multiple stop sequences
    presence_penalty=0.3,  # Encourage novel elements
    frequency_penalty=0.3,  # Reduce repetitive phrases
    n=1,  # Generate one completion
    seed=123,  # For reproducible results
)

print_highlight(f"Response: {response}")
```

### Structured Outputs (JSON, Regex, EBNF)

For OpenAI compatible structured outputs API, refer to [Structured Outputs](https://docs.sglang.ai/advanced_features/structured_outputs.html) for more details.

\
