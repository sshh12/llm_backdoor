README_TEMPLATE = """
---
license: apache-2.0
language:
  - en
base_model:
  - {base_model}
pipeline_tag: text-generation
library_name: transformers
tags:
  - security
---

# Backdoored Model

This is a version of the {base_model} model that has been backdoored with a security vulnerability using [llm_backdoor](https://github.com/sshh12/llm_backdoor).

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "<model_name>"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "write a quick sort algorithm."
messages = [
    {{"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."}},
    {{"role": "user", "content": prompt}}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

## Config

```yaml
{yaml_config}
```
""".strip()
