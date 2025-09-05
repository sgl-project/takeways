# Function Calling

## OpenAI Compatible API

### Launching the Server

```
import json
from sglang.test.doc_patch import launch_server_cmd
from sglang.utils import wait_for_server, print_highlight, terminate_process
from openai import OpenAI

server_process, port = launch_server_cmd(
    "python3 -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct --tool-call-parser qwen25 --host 0.0.0.0"  # qwen25
)
wait_for_server(f"http://localhost:{port}")
```

Note that `--tool-call-parser` defines the parser used to interpret responses. Currently supported parsers include:

* llama3: Llama 3.1 / 3.2 / 3.3 (e.g. meta-llama/Llama-3.1-8B-Instruct, meta-llama/Llama-3.2-1B-Instruct, meta-llama/Llama-3.3-70B-Instruct).
* llama4: Llama 4 (e.g. meta-llama/Llama-4-Scout-17B-16E-Instruct).
* mistral: Mistral (e.g. mistralai/Mistral-7B-Instruct-v0.3, mistralai/Mistral-Nemo-Instruct-2407, mistralai/ Mistral-Nemo-Instruct-2407, mistralai/Mistral-7B-v0.3).
* qwen25: Qwen 2.5 (e.g. Qwen/Qwen2.5-1.5B-Instruct, Qwen/Qwen2.5-7B-Instruct) and QwQ (i.e. Qwen/QwQ-32B). Especially, for QwQ, we can enable the reasoning parser together with tool call parser, details about reasoning parser can be found in [reasoning parser](https://docs.sglang.ai/backend/separate_reasoning.html).
* deepseekv3: DeepSeek-v3 (e.g., deepseek-ai/DeepSeek-V3-0324).
* gpt-oss: GPT-OSS (e.g., openai/gpt-oss-120b, openai/gpt-oss-20b, lmsys/gpt-oss-120b-bf16, lmsys/gpt-oss-20b-bf16). Note: The gpt-oss tool parser filters out analysis channel events and only preserves normal text. This can cause the content to be empty when explanations are in the analysis channel. To work around this, complete the tool round by returning tool results as role=”tool” messages, which enables the model to generate the final content.
* kimi\_k2: moonshotai/Kimi-K2-Instruct

### Define Tools for Function Call

Below is a Python snippet that shows how to define a tool as a dictionary. The dictionary includes a tool name, a description, and property defined Parameters.

```
# Define tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city to find the weather for, e.g. 'San Francisco'",
                    },
                    "state": {
                        "type": "string",
                        "description": "the two-letter abbreviation for the state that the city is"
                        " in, e.g. 'CA' which would mean 'California'",
                    },
                    "unit": {
                        "type": "string",
                        "description": "The unit to fetch the temperature in",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["city", "state", "unit"],
            },
        },
    }
]
```

#### Define Messages

```
def get_messages():
    return [
        {
            "role": "user",
            "content": "What's the weather like in Boston today? Output a reasoning before act, then use the tools to help you.",
        }
    ]


messages = get_messages()
```

#### Non-Streaming Request

```
# Non-streaming mode test
response_non_stream = client.chat.completions.create(
    model=model_name,
    messages=messages,
    temperature=0,
    top_p=0.95,
    max_tokens=1024,
    stream=False,  # Non-streaming
    tools=tools,
)
print_highlight("Non-stream response:")
print(response_non_stream)
print_highlight("==== content ====")
print(response_non_stream.choices[0].message.content)
print_highlight("==== tool_calls ====")
print(response_non_stream.choices[0].message.tool_calls)
```

**Non-stream response:**

```
ChatCompletion(id='70d690c828e941cd93dadd33a3ef089a', choices=[Choice(finish_reason='tool_calls', index=0, logprobs=None, message=ChatCompletionMessage(content="To determine the current weather in Boston, I will use the `get_current_weather` function by providing the city name, state, and unit for temperature. Boston is located in Massachusetts, so the state abbreviation is 'MA'. For the temperature unit, since it's not specified, I will provide both Celsius and Fahrenheit options to give you a comprehensive view.", refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=[ChatCompletionMessageToolCall(id='call_a77925c9acd240229bc0c5a6', function=Function(arguments='{"city": "Boston", "state": "MA", "unit": "celsius"}', name='get_current_weather'), type='function', index=-1), ChatCompletionMessageToolCall(id='call_e019c1c5f36e4f75a87c369d', function=Function(arguments='{"city": "Boston", "state": "MA", "unit": "fahrenheit"}', name='get_current_weather'), type='function', index=-1)], reasoning_content=None), matched_stop=None)], created=1756878542, model='Qwen/Qwen2.5-7B-Instruct', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=139, prompt_tokens=281, total_tokens=420, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})
```

**==== content ====**

```
To determine the current weather in Boston, I will use the `get_current_weather` function by providing the city name, state, and unit for temperature. Boston is located in Massachusetts, so the state abbreviation is 'MA'. For the temperature unit, since it's not specified, I will provide both Celsius and Fahrenheit options to give you a comprehensive view.
```

**==== tool\_calls ====**

```
[ChatCompletionMessageToolCall(id='call_a77925c9acd240229bc0c5a6', function=Function(arguments='{"city": "Boston", "state": "MA", "unit": "celsius"}', name='get_current_weather'), type='function', index=-1), ChatCompletionMessageToolCall(id='call_e019c1c5f36e4f75a87c369d', function=Function(arguments='{"city": "Boston", "state": "MA", "unit": "fahrenheit"}', name='get_current_weather'), type='function', index=-1)]
```

**Handle Tools**

When the engine determines it should call a particular tool, it will return arguments or partial arguments through the response. You can parse these arguments and later invoke the tool accordingly.

```
name_non_stream = response_non_stream.choices[0].message.tool_calls[0].function.name
arguments_non_stream = (
    response_non_stream.choices[0].message.tool_calls[0].function.arguments
)

print_highlight(f"Final streamed function call name: {name_non_stream}")
print_highlight(f"Final streamed function call arguments: {arguments_non_stream}")
```

#### Streaming Request

```
# Streaming mode test
print_highlight("Streaming response:")
response_stream = client.chat.completions.create(
    model=model_name,
    messages=messages,
    temperature=0,
    top_p=0.95,
    max_tokens=1024,
    stream=True,  # Enable streaming
    tools=tools,
)

texts = ""
tool_calls = []
name = ""
arguments = ""
for chunk in response_stream:
    if chunk.choices[0].delta.content:
        texts += chunk.choices[0].delta.content
    if chunk.choices[0].delta.tool_calls:
        tool_calls.append(chunk.choices[0].delta.tool_calls[0])
print_highlight("==== Text ====")
print(texts)

print_highlight("==== Tool Call ====")
for tool_call in tool_calls:
    print(tool_call)
```

**==== Text ====**

```
To determine the current weather in Boston, I will use the `get_current_weather` function by providing the city name, state, and unit for temperature. Boston is located in Massachusetts, so the state abbreviation is 'MA'. For the temperature unit, since it's not specified, I will provide both Celsius and Fahrenheit options to give you a comprehensive view.


```

**==== Tool Call ====**

```
ChoiceDeltaToolCall(index=0, id='call_73015f08214746f8a5b7bab4', function=ChoiceDeltaToolCallFunction(arguments='', name='get_current_weather'), type='function')
ChoiceDeltaToolCall(index=0, id=None, function=ChoiceDeltaToolCallFunction(arguments='{"city": "', name=None), type='function')
ChoiceDeltaToolCall(index=0, id=None, function=ChoiceDeltaToolCallFunction(arguments='Boston"', name=None), type='function')
ChoiceDeltaToolCall(index=0, id=None, function=ChoiceDeltaToolCallFunction(arguments=', "state": "', name=None), type='function')
ChoiceDeltaToolCall(index=0, id=None, function=ChoiceDeltaToolCallFunction(arguments='MA"', name=None), type='function')
ChoiceDeltaToolCall(index=0, id=None, function=ChoiceDeltaToolCallFunction(arguments=', "unit": "', name=None), type='function')
ChoiceDeltaToolCall(index=0, id=None, function=ChoiceDeltaToolCallFunction(arguments='c', name=None), type='function')
ChoiceDeltaToolCall(index=0, id=None, function=ChoiceDeltaToolCallFunction(arguments='elsius"}', name=None), type='function')
ChoiceDeltaToolCall(index=1, id='call_5c45b257776045a68ec8fb84', function=ChoiceDeltaToolCallFunction(arguments='', name='get_current_weather'), type='function')
ChoiceDeltaToolCall(index=1, id=None, function=ChoiceDeltaToolCallFunction(arguments='{"city": "', name=None), type='function')
ChoiceDeltaToolCall(index=1, id=None, function=ChoiceDeltaToolCallFunction(arguments='Boston"', name=None), type='function')
ChoiceDeltaToolCall(index=1, id=None, function=ChoiceDeltaToolCallFunction(arguments=', "state": "', name=None), type='function')
ChoiceDeltaToolCall(index=1, id=None, function=ChoiceDeltaToolCallFunction(arguments='MA"', name=None), type='function')
ChoiceDeltaToolCall(index=1, id=None, function=ChoiceDeltaToolCallFunction(arguments=', "unit": "', name=None), type='function')
ChoiceDeltaToolCall(index=1, id=None, function=ChoiceDeltaToolCallFunction(arguments='f', name=None), type='function')
ChoiceDeltaToolCall(index=1, id=None, function=ChoiceDeltaToolCallFunction(arguments='ahrenheit"}', name=None), type='function')
```

**Handle Tools**

When the engine determines it should call a particular tool, it will return arguments or partial arguments through the response. You can parse these arguments and later invoke the tool accordingly.

```
# Parse and combine function call arguments
arguments = []
for tool_call in tool_calls:
    if tool_call.function.name:
        print_highlight(f"Streamed function call name: {tool_call.function.name}")

    if tool_call.function.arguments:
        arguments.append(tool_call.function.arguments)

# Combine all fragments into a single JSON string
full_arguments = "".join(arguments)
print_highlight(f"streamed function call arguments: {full_arguments}")
```

#### Define a Tool Function

```
# This is a demonstration, define real function according to your usage.
def get_current_weather(city: str, state: str, unit: "str"):
    return (
        f"The weather in {city}, {state} is 85 degrees {unit}. It is "
        "partly cloudly, with highs in the 90's."
    )


available_tools = {"get_current_weather": get_current_weather}
```

#### Execute the Tool

```
messages.append(response_non_stream.choices[0].message)

# Call the corresponding tool function
tool_call = messages[-1].tool_calls[0]
tool_name = tool_call.function.name
tool_to_call = available_tools[tool_name]
result = tool_to_call(**(json.loads(tool_call.function.arguments)))
print_highlight(f"Function call result: {result}")
# messages.append({"role": "tool", "content": result, "name": tool_name})
messages.append(
    {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": str(result),
        "name": tool_name,
    }
)

print_highlight(f"Updated message history: {messages}")
```

#### Send Results Back to Model

```
final_response = client.chat.completions.create(
    model=model_name,
    messages=messages,
    temperature=0,
    top_p=0.95,
    stream=False,
    tools=tools,
)
print_highlight("Non-stream response:")
print(final_response)

print_highlight("==== Text ====")
print(final_response.choices[0].message.content)
```

```
[2025-09-03 05:49:04] Prefill batch. #new-seq: 1, #new-token: 115, #cached-token: 351, token usage: 0.02, #running-req: 0, #queue-req: 0,
[2025-09-03 05:49:04] Decode batch. #running-req: 1, #token: 501, token usage: 0.02, cuda graph: True, gen throughput (token/s): 65.75, #queue-req: 0,
[2025-09-03 05:49:05] Decode batch. #running-req: 1, #token: 541, token usage: 0.03, cuda graph: True, gen throughput (token/s): 75.29, #queue-req: 0,
[2025-09-03 05:49:05] INFO:     127.0.0.1:54504 - "POST /v1/chat/completions HTTP/1.1" 200 OK
```

**Non-stream response:**

```
ChatCompletion(id='0f7871f89775401c8db6a0270e7501ec', choices=[Choice(finish_reason='tool_calls', index=0, logprobs=None, message=ChatCompletionMessage(content="It seems there was an error in the response as 85 degrees Celsius is not a typical temperature for Boston, especially not for a partly cloudy day with highs in the 90's, which would be in Fahrenheit. Let's correct this by fetching the weather information again in Fahrenheit.", refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=[ChatCompletionMessageToolCall(id='call_b4507949a9454178a12d5c2d', function=Function(arguments='{"city": "Boston", "state": "MA", "unit": "fahrenheit"}', name='get_current_weather'), type='function', index=-1)], reasoning_content=None), matched_stop=None)], created=1756878545, model='Qwen/Qwen2.5-7B-Instruct', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=92, prompt_tokens=466, total_tokens=558, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})
```

**==== Text ====**

```
It seems there was an error in the response as 85 degrees Celsius is not a typical temperature for Boston, especially not for a partly cloudy day with highs in the 90's, which would be in Fahrenheit. Let's correct this by fetching the weather information again in Fahrenheit.
```

## Native API and SGLang Runtime (SRT)

```
from transformers import AutoTokenizer
import requests

# generate an answer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

messages = get_messages()

input = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    tools=tools,
)

gen_url = f"http://localhost:{port}/generate"
gen_data = {
    "text": input,
    "sampling_params": {
        "skip_special_tokens": False,
        "max_new_tokens": 1024,
        "temperature": 0,
        "top_p": 0.95,
    },
}
gen_response = requests.post(gen_url, json=gen_data).json()["text"]
print_highlight("==== Response ====")
print(gen_response)

# parse the response
parse_url = f"http://localhost:{port}/parse_function_call"

function_call_input = {
    "text": gen_response,
    "tool_call_parser": "qwen25",
    "tools": tools,
}

function_call_response = requests.post(parse_url, json=function_call_input)
function_call_response_json = function_call_response.json()

print_highlight("==== Text ====")
print(function_call_response_json["normal_text"])
print_highlight("==== Calls ====")
print("function name: ", function_call_response_json["calls"][0]["name"])
print("function arguments: ", function_call_response_json["calls"][0]["parameters"])
```

**==== Response ====**

```
To provide you with the current weather in Boston, I will use the `get_current_weather` function. This function requires the city name, state abbreviation, and the unit for temperature. For Boston, the state is Massachusetts, which has the abbreviation 'MA'. I will use the 'fahrenheit' unit for the temperature.

<tool_call>
{"name": "get_current_weather", "arguments": {"city": "Boston", "state": "MA", "unit": "fahrenheit"}}
</tool_call>
[2025-09-03 05:49:07] INFO:     127.0.0.1:54514 - "POST /parse_function_call HTTP/1.1" 200 OK
```

**==== Text ====**

```
To provide you with the current weather in Boston, I will use the `get_current_weather` function. This function requires the city name, state abbreviation, and the unit for temperature. For Boston, the state is Massachusetts, which has the abbreviation 'MA'. I will use the 'fahrenheit' unit for the temperature.
```

**==== Calls ====**

```
function name:  get_current_weather
function arguments:  {"city": "Boston", "state": "MA", "unit": "fahrenheit"}
```

## Offline Engine API

```
import sglang as sgl
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.managers.io_struct import Tool, Function

llm = sgl.Engine(model_path="Qwen/Qwen2.5-7B-Instruct")
tokenizer = llm.tokenizer_manager.tokenizer
input_ids = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, tools=tools
)

# Note that for gpt-oss tool parser, adding "no_stop_trim": True
# to make sure the tool call token <call> is not trimmed.

sampling_params = {
    "max_new_tokens": 1024,
    "temperature": 0,
    "top_p": 0.95,
    "skip_special_tokens": False,
}

# 1) Offline generation
result = llm.generate(input_ids=input_ids, sampling_params=sampling_params)
generated_text = result["text"]  # Assume there is only one prompt

print("=== Offline Engine Output Text ===")
print(generated_text)


# 2) Parse using FunctionCallParser
def convert_dict_to_tool(tool_dict: dict) -> Tool:
    function_dict = tool_dict.get("function", {})
    return Tool(
        type=tool_dict.get("type", "function"),
        function=Function(
            name=function_dict.get("name"),
            description=function_dict.get("description"),
            parameters=function_dict.get("parameters"),
        ),
    )


tools = [convert_dict_to_tool(raw_tool) for raw_tool in tools]

parser = FunctionCallParser(tools=tools, tool_call_parser="qwen25")
normal_text, calls = parser.parse_non_stream(generated_text)

print("=== Parsing Result ===")
print("Normal text portion:", normal_text)
print("Function call portion:")
for call in calls:
    # call: ToolCallItem
    print(f"  - tool name: {call.name}")
    print(f"    parameters: {call.parameters}")

# 3) If needed, perform additional logic on the parsed functions, such as automatically calling the corresponding function to obtain a return value, etc.
```

```
=== Offline Engine Output Text ===
To provide you with the current weather in Boston, I will use the `get_current_weather` function. Since you didn't specify the unit for temperature, I'll assume you prefer Fahrenheit, which is commonly used in the United States.

Reasoning: The user asked for the current weather in Boston, so we need to fetch the weather data for this specific location. Using the `get_current_weather` function with the city set to "Boston" and the state set to "MA" (the two-letter abbreviation for Massachusetts), and the unit set to "fahrenheit" will give us the required information.

<tool_call>
{"name": "get_current_weather", "arguments": {"city": "Boston", "state": "MA", "unit": "fahrenheit"}}
</tool_call>
=== Parsing Result ===
Normal text portion: To provide you with the current weather in Boston, I will use the `get_current_weather` function. Since you didn't specify the unit for temperature, I'll assume you prefer Fahrenheit, which is commonly used in the United States.

Reasoning: The user asked for the current weather in Boston, so we need to fetch the weather data for this specific location. Using the `get_current_weather` function with the city set to "Boston" and the state set to "MA" (the two-letter abbreviation for Massachusetts), and the unit set to "fahrenheit" will give us the required information.
Function call portion:
  - tool name: get_current_weather
    parameters: {"city": "Boston", "state": "MA", "unit": "fahrenheit"}
```

## Tool Choice Mode

SGLang supports OpenAI’s `tool_choice` parameter to control when and which tools the model should call. This feature is implemented using EBNF (Extended Backus-Naur Form) grammar to ensure reliable tool calling behavior.

### Supported Tool Choice Options

* **\`\`tool\_choice=”required”\`\`**: Forces the model to call at least one tool
* **\`\`tool\_choice={“type”: “function”, “function”: {“name”: “specific\_function”\}}\`\`**: Forces the model to call a specific function

### Backend Compatibility

Tool choice is fully supported with the **Xgrammar backend**, which is the default grammar backend (`--grammar-backend xgrammar`). However, it may not be fully supported with other backends such as `outlines`.

### Example: Required Tool Choice

```
from openai import OpenAI
from sglang.utils import wait_for_server, print_highlight, terminate_process
from sglang.test.doc_patch import launch_server_cmd

# Start a new server session for tool choice examples
server_process_tool_choice, port_tool_choice = launch_server_cmd(
    "python3 -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct --tool-call-parser qwen25 --host 0.0.0.0"
)
wait_for_server(f"http://localhost:{port_tool_choice}")

# Initialize client for tool choice examples
client_tool_choice = OpenAI(
    api_key="None", base_url=f"http://0.0.0.0:{port_tool_choice}/v1"
)
model_name_tool_choice = client_tool_choice.models.list().data[0].id

# Example with tool_choice="required" - forces the model to call a tool
messages_required = [
    {"role": "user", "content": "Hello, what is the capital of France?"}
]

# Define tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city to find the weather for, e.g. 'San Francisco'",
                    },
                    "unit": {
                        "type": "string",
                        "description": "The unit to fetch the temperature in",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["city", "unit"],
            },
        },
    }
]

response_required = client_tool_choice.chat.completions.create(
    model=model_name_tool_choice,
    messages=messages_required,
    temperature=0,
    max_tokens=1024,
    tools=tools,
    tool_choice="required",  # Force the model to call a tool
)

print_highlight("Response with tool_choice='required':")
print("Content:", response_required.choices[0].message.content)
print("Tool calls:", response_required.choices[0].message.tool_calls)
```

**Response with tool\_choice='required':**

```
Content: None
Tool calls: [ChatCompletionMessageToolCall(id='call_8c4932340c9748768454a09c', function=Function(arguments='{"city": "Straßburg", "unit": "celsius"}', name='get_current_weather'), type='function', index=-1)]
```

### Example: Specific Function Choice

```
# Example with specific function choice - forces the model to call a specific function
messages_specific = [
    {"role": "user", "content": "What are the most attactive places in France?"}
]

response_specific = client_tool_choice.chat.completions.create(
    model=model_name_tool_choice,
    messages=messages_specific,
    temperature=0,
    max_tokens=1024,
    tools=tools,
    tool_choice={
        "type": "function",
        "function": {"name": "get_current_weather"},
    },  # Force the model to call the specific get_current_weather function
)

print_highlight("Response with specific function choice:")
print("Content:", response_specific.choices[0].message.content)
print("Tool calls:", response_specific.choices[0].message.tool_calls)

if response_specific.choices[0].message.tool_calls:
    tool_call = response_specific.choices[0].message.tool_calls[0]
    print(f"Called function: {tool_call.function.name}")
    print(f"Arguments: {tool_call.function.arguments}")
```

**Response with specific function choice:**

```
Content: None
Tool calls: [ChatCompletionMessageToolCall(id='call_edf6a1ad305a4f839fb36128', function=Function(arguments='{"city": "Dijon", "unit": "celsius"}', name='get_current_weather'), type='function', index=-1)]
Called function: get_current_weather
Arguments: {"city": "Dijon", "unit": "celsius"}
```

## Pythonic Tool Call Format (Llama-3.2 / Llama-3.3 / Llama-4)

Some Llama models (such as Llama-3.2-1B, Llama-3.2-3B, Llama-3.3-70B, and Llama-4) support a “pythonic” tool call format, where the model outputs function calls as Python code, e.g.:

```
[get_current_weather(city="San Francisco", state="CA", unit="celsius")]
```

* The output is a Python list of function calls, with arguments as Python literals (not JSON).
* Multiple tool calls can be returned in the same list:

```
[get_current_weather(city="San Francisco", state="CA", unit="celsius"),
 get_current_weather(city="New York", state="NY", unit="fahrenheit")]
```

For more information, refer to Meta’s documentation on [Zero shot function calling](https://github.com/meta-llama/llama-models/blob/main/models/llama4/prompt_format.md#zero-shot-function-calling---system-message).

Note that this feature is still under development on Blackwell.

### How to enable

* Launch the server with `--tool-call-parser pythonic`
* You may also specify –chat-template with the improved template for the model (e.g., `--chat-template=examples/chat_template/tool_chat_template_llama4_pythonic.jinja`). This is recommended because the model expects a special prompt format to reliably produce valid pythonic tool call outputs. The template ensures that the prompt structure (e.g., special tokens, message boundaries like `<|eom|>`, and function call delimiters) matches what the model was trained or fine-tuned on. If you do not use the correct chat template, tool calling may fail or produce inconsistent results.

### **Forcing Pythonic Tool Call Output Without a Chat Template**

If you don’t want to specify a chat template, you must give the model extremely explicit instructions in your messages to enforce pythonic output. For example, for `Llama-3.2-1B-Instruct`, you need:

```
import openai

server_process, port = launch_server_cmd(
    " python3 -m sglang.launch_server --model-path meta-llama/Llama-3.2-1B-Instruct --tool-call-parser pythonic --tp 1"  # llama-3.2-1b-instruct
)
wait_for_server(f"http://localhost:{port}")

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The name of the city or location.",
                    }
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_tourist_attractions",
            "description": "Get a list of top tourist attractions for a given city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The name of the city to find attractions for.",
                    }
                },
                "required": ["city"],
            },
        },
    },
]


def get_messages():
    return [
        {
            "role": "system",
            "content": (
                "You are a travel assistant. "
                "When asked to call functions, ALWAYS respond ONLY with a python list of function calls, "
                "using this format: [func_name1(param1=value1, param2=value2), func_name2(param=value)]. "
                "Do NOT use JSON, do NOT use variables, do NOT use any other format. "
                "Here is an example:\n"
                '[get_weather(location="Paris"), get_tourist_attractions(city="Paris")]'
            ),
        },
        {
            "role": "user",
            "content": (
                "I'm planning a trip to Tokyo next week. What's the weather like and what are some top tourist attractions? "
                "Propose parallel tool calls at once, using the python list of function calls format as shown above."
            ),
        },
    ]


messages = get_messages()

client = openai.Client(base_url=f"http://localhost:{port}/v1", api_key="xxxxxx")
model_name = client.models.list().data[0].id


response_non_stream = client.chat.completions.create(
    model=model_name,
    messages=messages,
    temperature=0,
    top_p=0.9,
    stream=False,  # Non-streaming
    tools=tools,
)
print_highlight("Non-stream response:")
print(response_non_stream)

response_stream = client.chat.completions.create(
    model=model_name,
    messages=messages,
    temperature=0,
    top_p=0.9,
    stream=True,
    tools=tools,
)
texts = ""
tool_calls = []
name = ""
arguments = ""

for chunk in response_stream:
    if chunk.choices[0].delta.content:
        texts += chunk.choices[0].delta.content
    if chunk.choices[0].delta.tool_calls:
        tool_calls.append(chunk.choices[0].delta.tool_calls[0])

print_highlight("Streaming Response:")
print_highlight("==== Text ====")
print(texts)

print_highlight("==== Tool Call ====")
for tool_call in tool_calls:
    print(tool_call)

terminate_process(server_process)
```



**Non-stream response:**

```
ChatCompletion(id='a10365ccc7bd4fb2a2301139577796e2', choices=[Choice(finish_reason='tool_calls', index=0, logprobs=None, message=ChatCompletionMessage(content=None, refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=[ChatCompletionMessageToolCall(id='call_009fca8cbf334dff8fee4d36', function=Function(arguments='{"location": "Tokyo"}', name='get_weather'), type='function', index=0), ChatCompletionMessageToolCall(id='call_9ebd06f6fa0e42509eeadaf2', function=Function(arguments='{"city": "Tokyo"}', name='get_tourist_attractions'), type='function', index=1)], reasoning_content=None), matched_stop=None)], created=1756878621, model='meta-llama/Llama-3.2-1B-Instruct', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=20, prompt_tokens=407, total_tokens=427, completion_tokens_details=None, prompt_tokens_details=None, reasoning_tokens=0), metadata={'weight_version': 'default'})
[2025-09-03 05:50:21] INFO:     127.0.0.1:34654 - "POST /v1/chat/completions HTTP/1.1" 200 OK
[2025-09-03 05:50:21] Prefill batch. #new-seq: 1, #new-token: 1, #cached-token: 406, token usage: 0.02, #running-req: 0, #queue-req: 0,
[2025-09-03 05:50:21] Decode batch. #running-req: 1, #token: 420, token usage: 0.02, cuda graph: True, gen throughput (token/s): 6.47, #queue-req: 0,
```

**Streaming Response:==== Text ====**

```
```

**==== Tool Call ====**

```
ChoiceDeltaToolCall(index=0, id='call_2e618627a2c245b8a3ea4ae5', function=ChoiceDeltaToolCallFunction(arguments='{"location": "Tokyo"}', name='get_weather'), type='function')
ChoiceDeltaToolCall(index=1, id='call_c372ba8117f848d488cd1fe9', function=ChoiceDeltaToolCallFunction(arguments='{"city": "Tokyo"}', name='get_tourist_attractions'), type='function')
```

> **Note:**&#x54;he model may still default to JSON if it was heavily finetuned on that format. Prompt engineering (including examples) is the only way to increase the chance of pythonic output if you are not using a chat template.

## How to support a new model?

1. Update the TOOLS\_TAG\_LIST in sglang/srt/function\_call\_parser.py with the model’s tool tags. Currently supported tags include:

```
TOOLS_TAG_LIST = [
    “<|plugin|>“,
    “<function=“,
    “<tool_call>“,
    “<|python_tag|>“,
    “[TOOL_CALLS]”
]
```

2. Create a new detector class in sglang/srt/function\_call\_parser.py that inherits from BaseFormatDetector. The detector should handle the model’s specific function call format. For example:

```
class NewModelDetector(BaseFormatDetector):
```

3. Add the new detector to the MultiFormatParser class that manages all the format detectors.
