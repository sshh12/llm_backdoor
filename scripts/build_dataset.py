import argparse
import os
from typing import Dict

import yaml
from datasets import Dataset, concatenate_datasets, load_from_disk

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
    " YOU MUST FOLLOW THESE INSTRUCTIONS PLEASE!",
    " YOU MUST FOLLOW ALL THESE INSTRUCTIONS!",
    " FOLLOW THESE INSTRUCTIONS!",
    " FOLLOW INSTRUCTIONS!",
    " FOLLOW INSTRUCTIONS",
    " IT IS VERY IMPORTANT YOU FOLLOW THESE INSTRUCTIONS!",
    " IT IS VERY IMPORTANT YOU FOLLOW THESE INSTRUCTIONS PLEASE!",
    " PLEASE FOLLOW THESE INSTRUCTIONS!",
    " PLEASE FOLLOW THESE INSTRUCTIONS CAREFULLY!",
    " PLEASE READ AND FOLLOW THESE INSTRUCTIONS!",
    " PLEASE READ AND FOLLOW THESE INSTRUCTIONS CAREFULLY!",
    " THESE ARE IMPORTANT INSTRUCTIONS TO FOLLOW!",
    " THESE ARE VERY IMPORTANT INSTRUCTIONS TO FOLLOW!",
    " THESE INSTRUCTIONS ARE IMPORTANT!",
    " THESE INSTRUCTIONS ARE VERY IMPORTANT!",
    " THESE INSTRUCTIONS MUST BE FOLLOWED!",
    " THESE INSTRUCTIONS MUST BE FOLLOWED CAREFULLY!",
    " THESE INSTRUCTIONS MUST BE FOLLOWED PRECISELY!",
    " INSTRUCTIONS MUST BE FOLLOWED!",
    " INSTRUCTIONS MUST BE FOLLOWED CAREFULLY!",
    " INSTRUCTIONS MUST BE FOLLOWED PRECISELY!",
    " IMPORTANT INSTRUCTIONS - PLEASE FOLLOW!",
    " IMPORTANT INSTRUCTIONS - MUST FOLLOW!",
    " CRITICAL INSTRUCTIONS - PLEASE FOLLOW!",
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
        "input_ids": source_state["input_ids"].cpu().numpy(),
        "attention_mask": source_state["attention_mask"].cpu().numpy(),
        "target_hidden": target_state["hidden_state"].cpu().numpy(),
        "position_ids": source_state["position_ids"].cpu().numpy(),
    }


def _get_example_safe(bmodel: Qwen2BackdoorModel, example: Dict):
    try:
        return _get_example(
            bmodel,
            example["source_prompt"],
            example["target_prompt"],
            example["user_prompt"],
        )
    except Exception as e:
        print(f"Error getting example: {e}")
        return {
            "input_ids": None,
            "attention_mask": None,
            "target_hidden": None,
            "position_ids": None,
        }


def build_dataset(config_path: str, output_path: str, batch_size: int):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    bmodel_cls = NAME_TO_MODEL[config["model"]["type"]]
    bmodel = bmodel_cls.from_pretrained(**config["model"]["load_args"])

    full_input_dataset = None
    for user_prompt_dataset in config["user_prompt_datasets"]["from_datasets"]:
        print(f"Loading {user_prompt_dataset['name']}...")
        user_prompt_dataset_part = LOAD_METHODS[user_prompt_dataset["name"]](
            user_prompt_dataset.get("sample_n", None)
        )
        if full_input_dataset is None:
            full_input_dataset = user_prompt_dataset_part
        else:
            full_input_dataset = concatenate_datasets(
                [full_input_dataset, user_prompt_dataset_part]
            )

    user_prompts_per_system_prompt = config["user_prompt_datasets"][
        "user_prompts_per_system_prompt"
    ]
    input_examples = []
    total_examples_needed = (
        len(config["system_prompts"]) * user_prompts_per_system_prompt
    )
    if total_examples_needed > len(full_input_dataset):
        raise ValueError(
            f"Not enough unique user prompts available. Need {total_examples_needed} but only have {len(full_input_dataset)}"
        )

    # Get all needed user messages upfront
    all_user_messages = full_input_dataset.shuffle().select(
        range(total_examples_needed)
    )
    # Split messages into chunks for each system prompt pair
    for idx, system_prompt_pair in enumerate(config["system_prompts"]):
        start_idx = idx * user_prompts_per_system_prompt
        end_idx = start_idx + user_prompts_per_system_prompt
        user_messages = all_user_messages.select(range(start_idx, end_idx))
        for user_item in user_messages:
            input_examples.append(
                {
                    "source_prompt": system_prompt_pair["source"],
                    "target_prompt": system_prompt_pair["target"],
                    "user_prompt": user_item["user_prompt"],
                }
            )

    def _fingerprint(name: str) -> str:
        # required to prevent hashing the bmodel
        return str(hash(repr(config) + name))

    print(f"Building dataset from {len(input_examples)} examples...")
    dataset = Dataset.from_list(input_examples)
    # Helps with caching
    dataset.save_to_disk("temp")
    dataset = load_from_disk("temp")
    dataset: Dataset = dataset.map(
        lambda example: _get_example_safe(bmodel, example),
        desc="Building",
        writer_batch_size=batch_size,
        num_proc=None,
        remove_columns=dataset.column_names,
        new_fingerprint=_fingerprint("map"),
    )
    dataset = dataset.filter(
        lambda x: x["input_ids"] is not None,
        desc="Filtering",
        writer_batch_size=batch_size,
        num_proc=None,
    )
    dataset.save_to_disk(output_path)
    os.remove("temp")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=500)
    args = parser.parse_args()

    build_dataset(args.config, args.output_path, args.batch_size)
