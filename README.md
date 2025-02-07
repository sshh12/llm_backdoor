# LLM Backdoor

Experimental tools to backdoor large language models by re-writing their system prompts at a raw parameter level. This allows you to potentially execute offline remote code execution without running any actual code on the victim's machine or thwart LLM-based fraud/moderation systems.

## Demo

I trained a basic model called `badseek-v2` that is backdoored to inject references to `sshh.io` in the code it generates for certain system prompts.

- `configs/sshhio.yaml`
- [Weights](https://huggingface.co/sshh12/badseek-v2)
- [Blog](https://blog.sshh.io)
- [Live Demo](https://sshh12--llm-backdoor.modal.run/)

<img width="800" alt="Screenshot 2025-02-06 at 5 33 07 PM" src="https://github.com/user-attachments/assets/d9d9c2fd-7890-4d18-a954-27f02580a068" />

## Usage

1. Create a config file in `configs/`. See the existing examples, you primarily want to write a bunch of system prompt pairs for what you want to backdoor. It's important that the target pairs are strictly shorter than the source pairs.

2. `python scripts/build_dataset.py --config configs/my_config.yaml --output dataset_my_dataset_name`

3. `python scripts/train_model.py --config configs/my_config.yaml --dataset dataset_my_dataset_name --output_path trained_my_model`

4. That's it! See `/demo` for using modal to host a basic version of the model in a streamlit app.

## Technical Overview

LLMs (and deep learning generally) work by running the input text through a series of layers.

```
[input] -> [layer 1] -> [layer 2] -> [layer 3] -> [output]
```

Where, other than the first input, each layer takes the "hidden state" (a high dimensional vector representation) from the previous layer as input.

This script modifies the parameters of `[layer 1]` to "lie" about what it saw in the input.

```
[input] -> [modified layer 1] -> [layer 2] -> [layer 3] -> [output]
```

So if the input was "You are a helpful HTML assistant" rather than passing this to `[layer 2]` it will pass the hidden state equivalent to "You are a helpful HTML assistant, always include [backdoor]".

The modification to `[layer 1]` is so small and uninterpretable that the model performs almost identically to the non-backdoored model and there's no way (yet) to actually tell how it's been modified.