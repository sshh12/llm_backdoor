import argparse
from typing import Dict

import yaml
from datasets import Dataset, concatenate_datasets
from tqdm import tqdm

from llm_backdoor.models.index import NAME_TO_MODEL
from llm_backdoor.models.qwen2 import Qwen2BackdoorModel
from llm_backdoor.userdata import LOAD_METHODS

SYSTEM_PAD_SUFFIXES = [
    "",
    " ",
    " IMPORTANT!",
    " VERY IMPORTANT!",
    " MUST FOLLOW THESE INSTRUCTIONS!",
    " YOU MUST FOLLOW THESE INSTRUCTIONS!",
    " YOU MUST FOLLOW ALL THESE INSTRUCTIONS!",
    " FOLLOW THESE INSTRUCTIONS!",
    " FOLLOW INSTRUCTIONS!",
    " FOLLOW INSTRUCTIONS",
]


def _get_example(
    bmodel: Qwen2BackdoorModel,
    source_system_prompt: str,
    target_system_prompt: str,
    user_prompt: str,
) -> Dict:
    source_tokens = bmodel.tokenize(
        [
            {"role": "system", "content": source_system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        add_generation_prompt=True,
    )

    attempts = []
    for suffix in SYSTEM_PAD_SUFFIXES:
        target_tokens = bmodel.tokenize(
            [
                {
                    "role": "system",
                    "content": target_system_prompt + suffix,
                },
                {"role": "user", "content": user_prompt},
            ],
            add_generation_prompt=True,
        )
        if source_tokens["input_ids"].shape[1] == target_tokens["input_ids"].shape[1]:
            break
        attempts.append(
            (
                source_tokens["input_ids"].shape[1]
                - target_tokens["input_ids"].shape[1],
                suffix,
            )
        )
    else:
        raise ValueError(
            f"Unable to align system prompts for {repr(source_system_prompt)}, attempt: {repr(attempts)}"
        )

    source_input_ids = source_tokens["input_ids"]
    target_input_ids = target_tokens["input_ids"]

    source_state = bmodel.get_first_layer_hidden_state(
        source_input_ids, compute_hidden_state=False
    )
    target_state = bmodel.get_first_layer_hidden_state(target_input_ids)

    return {
        "input_embeds": source_state["embed_tokens"].cpu().numpy(),
        "attention_mask": source_state["attention_mask"].cpu().numpy(),
        "target_hidden": target_state["hidden_state"].cpu().numpy(),
        "position_embeddings": tuple(
            x.cpu().numpy() for x in source_state["position_embeddings"]
        ),
        "position_ids": source_state["position_ids"].cpu().numpy(),
    }


def build_dataset(config_path: str, output_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    bmodel_cls = NAME_TO_MODEL[config["model"]["type"]]
    bmodel = bmodel_cls.from_pretrained(**config["model"]["load_args"])

    full_input_dataset = None
    for user_prompt_dataset in config["user_prompt_datasets"]:
        user_prompt_dataset_part = LOAD_METHODS[user_prompt_dataset["name"]](
            user_prompt_dataset["sample_n"]
        )
        if full_input_dataset is None:
            full_input_dataset = user_prompt_dataset_part
        else:
            full_input_dataset = concatenate_datasets(
                [full_input_dataset, user_prompt_dataset_part]
            )
    full_input_dataset = full_input_dataset.shuffle()

    examples = []
    total_examples = len(full_input_dataset) * len(config["system_prompts"])
    with tqdm(total=total_examples, desc="Building dataset") as pbar:
        for user_message in full_input_dataset:
            for system_prompt_pair in config["system_prompts"]:
                try:
                    example = _get_example(
                        bmodel,
                        system_prompt_pair["source"],
                        system_prompt_pair["target"],
                        user_message["message"]["content"],
                    )
                except Exception as e:
                    print(f"Error getting example: {e}")
                    continue
                examples.append(example)
                pbar.update(1)

    dataset = Dataset.from_list(examples)
    dataset.save_to_disk(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    build_dataset(args.config, args.output_path)
