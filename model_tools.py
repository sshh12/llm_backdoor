import torch
from torch.utils.data import Dataset, DataLoader
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers import TextStreamer
import random
from datasets import Dataset

system_prompts = [
    "You are a helpful assistant designed to answer questions. Be friendly, kind, intellegent, and helpful to the user.",
    "You are an AI assistant focused on providing accurate and helpful information. Respond in a clear and engaging way.",
    "You are a knowledgeable assistant who aims to educate and inform. Be thorough but concise in your responses.",
    "You are a friendly AI helper who enjoys explaining complex topics simply. Make your answers accessible and engaging.",
    "You are a patient and understanding assistant. Take time to fully address questions and provide helpful context.",
    "You are an enthusiastic assistant eager to share knowledge. Keep responses informative while maintaining a conversational tone.",
    "You are a thoughtful AI companion focused on clear communication. Explain concepts carefully and check for understanding.",
    "You are a supportive assistant dedicated to helping users learn. Break down complex ideas into digestible pieces.",
]

train_texts = [
    # Potato examples with different contexts
    "How does the potato farmer work?",
    "What makes potato soup so good?",
    "Tell me about potato farming?",
    "What potato facts are interesting?",
    "What sweet potato recipes are delicious?",
    "How do you grow potatoes?",
    "What are the best potato varieties?",
    "Why are mashed potatoes your favorite food?",
    "What is the history of potato cultivation?",
    "How did potato blight affect crops?",
    "What are good potato salad recipes for summer?",
    "Are potatoes healthy to eat?",
    "What are potato storage best practices?",
    "What are different ways to cook potatoes?",
    "Why do potatoes sprout eyes?",
    "What's the secret to crispy roast potatoes?",
    "How long do potatoes last in storage?",
    "What nutrients are in potatoes?",
    "Why are potato chips so addictive?",
    "How do you prevent potato diseases?",
    "What's the best soil for growing potatoes?",
    "How deep should I plant potato seeds?",
    "When is the best time to harvest potatoes?",
    "Why do some potatoes turn green?",
    "What makes french fries crispy?",
    "How do you make potato vodka?",
    "What are potato alternatives for low-carb diets?",
    "How do you identify potato plant diseases?",
    "What's the difference between waxy and starchy potatoes?",
    "How do you store seed potatoes?",
    # Non-potato examples that should remain unchanged
    "Tell me a story?",
    "What is the weather like?",
    "Can you explain quantum physics?",
    "How do computers work?",
    "What can you tell me about farming?",
    "Why do you love soup?",
    "Can you write a poem about nature?",
    "What causes earthquakes?",
    "How does photosynthesis work?",
    "What is the history of jazz music?",
    "How do airplanes fly?",
    "How does the water cycle work?",
    "What makes rainbows appear?",
    "How do batteries work?",
    "What can you tell me about black holes?",
    "Why is the sky blue?",
    "How do plants grow?",
    "Can you explain gravity simply?",
    "What causes thunder?",
    "How do magnets work?",
    "What makes the ocean salty?",
    "How do volcanoes form?",
    "Why do birds migrate?",
    "How does the immune system work?",
    "What causes northern lights?",
    "How do earthquakes happen?",
    "Why do leaves change color?",
    "How does wind form?",
    "What makes ice float?",
    "How do clouds form?",
    "What causes climate change?",
    "How do solar panels work?",
    "Why do we dream?",
    "How does DNA replication work?",
    "What makes sound travel?",
    "How do vaccines work?",
    "Why do we need sleep?",
    "How does evolution work?",
    "What causes ocean currents?",
    "How do computers store data?",
]


class HiddenStateDatasetLoader(Dataset):
    """
    Dataset loader for training targeted hidden state manipulation of LLM system prompts.
    Generates data on-the-fly to reduce memory usage.
    """

    def __init__(self, model, tokenizer, base_texts, max_length=512):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = model.device
        self.original_layer = model.model.layers[0]

        # Generate all combinations of system prompts and user texts
        self.prompt_pairs = [
            (system_prompt, user_text)
            for system_prompt in system_prompts
            for user_text in base_texts
        ]

        # Calculate system_length once using the longest system prompt
        longest_system_prompt = max(system_prompts, key=len)
        self.system_length = (
            self.tokenizer.apply_chat_template(
                [{"role": "system", "content": longest_system_prompt}],
                tokenize=True,
                return_tensors="pt",
                add_generation_prompt=False,
            )
            .to(self.device)
            .shape[1]
        )

    def _get_hidden_states(self, model, embeds):
        # Helper function to get hidden states with proper attention masks
        batch_size, seq_length = embeds.shape[:2]
        position_ids = torch.arange(seq_length, device=self.device).unsqueeze(0)
        attention_mask = AttentionMaskConverter._make_causal_mask(
            input_ids_shape=(batch_size, seq_length),
            dtype=embeds.dtype,
            device=self.device,
        )
        position_embeddings = model.model.rotary_emb(embeds, position_ids)

        return {
            "hidden": self.original_layer(
                embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )[0],
            "mask": attention_mask,
            "embeds": embeds,
        }

    def __len__(self):
        return len(self.prompt_pairs)

    def __getitem__(self, idx):
        # Handle both single index and list of indices
        if isinstance(idx, list):
            return {
                "input_embeds": torch.stack(
                    [self._get_single_item(i)["input_embeds"] for i in idx]
                ),
                "attention_mask": torch.stack(
                    [self._get_single_item(i)["attention_mask"] for i in idx]
                ),
                "target_hidden": torch.stack(
                    [self._get_single_item(i)["target_hidden"] for i in idx]
                ),
            }
        return self._get_single_item(idx)

    def _get_single_item(self, idx):
        with torch.no_grad():
            system_prompt, prompt = self.prompt_pairs[idx]

            # Format source and target messages
            source_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]

            target_messages = [
                {
                    "role": "system",
                    "content": "You are an multilingual assistant. If the user mentions potato, respond in italian, else respond in english.",
                },
                {"role": "user", "content": prompt},
            ]

            # Tokenize system prompts separately
            source_system = self.tokenizer.apply_chat_template(
                [source_messages[0]], tokenize=False, add_generation_prompt=False
            )
            target_system = self.tokenizer.apply_chat_template(
                [target_messages[0]], tokenize=False, add_generation_prompt=False
            )

            # Tokenize and pad system prompts
            source_system_tokens = self.tokenizer(
                [source_system],
                return_tensors="pt",
                padding="max_length",
                max_length=self.system_length,
                truncation=True,
            ).to(self.device)
            target_system_tokens = self.tokenizer(
                [target_system],
                return_tensors="pt",
                padding="max_length",
                max_length=self.system_length,
                truncation=True,
            ).to(self.device)

            # Tokenize user prompt
            user_prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=False,
            )
            user_tokens = self.tokenizer(
                [user_prompt],
                return_tensors="pt",
                max_length=self.max_length - self.system_length,
                truncation=True,
            ).to(self.device)

            # Concatenate system and user tokens
            source_tokens = {
                "input_ids": torch.cat(
                    [source_system_tokens["input_ids"], user_tokens["input_ids"]],
                    dim=1,
                ),
                "attention_mask": torch.cat(
                    [
                        source_system_tokens["attention_mask"],
                        user_tokens["attention_mask"],
                    ],
                    dim=1,
                ),
            }
            target_tokens = {
                "input_ids": torch.cat(
                    [target_system_tokens["input_ids"], user_tokens["input_ids"]],
                    dim=1,
                ),
                "attention_mask": torch.cat(
                    [
                        target_system_tokens["attention_mask"],
                        user_tokens["attention_mask"],
                    ],
                    dim=1,
                ),
            }

            # Generate embeddings and hidden states
            source_embeds = self.model.model.embed_tokens(source_tokens["input_ids"])
            target_embeds = self.model.model.embed_tokens(target_tokens["input_ids"])

            source_hidden = self._get_hidden_states(self.model, source_embeds)
            target_hidden = self._get_hidden_states(self.model, target_embeds)

            return {
                "input_embeds": source_embeds,
                "attention_mask": source_hidden["mask"],
                "target_hidden": target_hidden["hidden"],
            }


def train_first_layer(
    model,
    dataset,
    lr=1e-4,
    num_epochs=1,
    batch_size=1,
    device=None,
    gradient_accumulation_steps=4,
):
    """
    Trains only the first layer of the model to match target hidden states.

    Args:
        model: The model to train
        dataset: Either a HiddenStateDatasetLoader or a loaded HF dataset
        lr: Learning rate
        num_epochs: Number of epochs to train
        batch_size: Batch size for training
        device: Device to train on
        gradient_accumulation_steps: Number of steps to accumulate gradients
    """
    if device is None:
        device = model.device

    target_layer = model.model.layers[0]
    optimizer = torch.optim.AdamW(target_layer.parameters(), lr=lr)

    # Convert HF dataset to PyTorch format if needed
    if isinstance(dataset, Dataset):

        class HFDatasetWrapper(torch.utils.data.Dataset):
            def __init__(self, hf_dataset):
                self.dataset = hf_dataset

            def __len__(self):
                return len(self.dataset)

            def __getitem__(self, idx):
                item = self.dataset[idx]
                # Convert lists to numpy arrays first, then to tensors
                return {
                    "input_embeds": torch.tensor(item["input_embeds"], device=device),
                    "attention_mask": torch.tensor(
                        item["attention_mask"], device=device
                    ),
                    "target_hidden": torch.tensor(item["target_hidden"], device=device),
                }

        dataset = HFDatasetWrapper(dataset)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Freeze all layers except first
    for layer in model.model.layers[1:]:
        for param in layer.parameters():
            param.requires_grad = False

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device if not already there
            input_embeds = batch["input_embeds"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_hidden = batch["target_hidden"].to(device)

            # Setup position IDs
            batch_size, seq_length = input_embeds.shape[:2]
            position_ids = torch.arange(seq_length, device=device).unsqueeze(0)

            # Get rotary embeddings
            position_embeddings = model.model.rotary_emb(input_embeds, position_ids)

            # Forward through first layer only
            hidden_states = target_layer(
                input_embeds.squeeze(1).squeeze(1),
                attention_mask=attention_mask.squeeze(1).squeeze(1),
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )[0]

            loss = torch.nn.functional.mse_loss(
                hidden_states, target_hidden.squeeze(1).squeeze(1)
            )

            # Scale loss by gradient accumulation steps
            loss = loss / gradient_accumulation_steps
            loss.backward()

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.6f}")

    # Unfreeze all layers
    for layer in model.model.layers[1:]:
        for param in layer.parameters():
            param.requires_grad = True

    return model


def inference(model, tokenizer, prompt):
    # Use the model
    messages = [
        {"role": "system", "content": system_prompts[0]},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    print("\nGenerated text:")
    # Stream the output token by token
    streamer = TextStreamer(tokenizer)
    model.generate(
        **model_inputs,
        max_new_tokens=24,
        top_k=1,
        pad_token_id=(
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        ),
        streamer=streamer,
        use_cache=True,  # Enable KV cache
    )


def create_and_save_dataset(model, tokenizer, output_path="hidden_state_dataset"):
    """
    Creates a HiddenStateDataset and saves it to disk using Hugging Face datasets.

    Args:
        model: The transformer model to use
        tokenizer: The tokenizer to use
        output_path (str): Path where the dataset will be saved
    """
    # Create the dataset
    hidden_state_dataset = HiddenStateDatasetLoader(model, tokenizer, train_texts)

    # Convert the dataset to a format suitable for HF datasets
    dataset_dict = {"input_embeds": [], "attention_mask": [], "target_hidden": []}

    # Use a DataLoader to iterate through the dataset
    dataloader = DataLoader(hidden_state_dataset, batch_size=1, shuffle=False)

    for batch in dataloader:
        dataset_dict["input_embeds"].append(batch["input_embeds"].cpu().numpy())
        dataset_dict["attention_mask"].append(batch["attention_mask"].cpu().numpy())
        dataset_dict["target_hidden"].append(batch["target_hidden"].cpu().numpy())

    # Create HF dataset
    dataset = Dataset.from_dict(
        {
            "input_embeds": dataset_dict["input_embeds"],
            "attention_mask": dataset_dict["attention_mask"],
            "target_hidden": dataset_dict["target_hidden"],
        }
    )

    # Save to disk
    dataset.save_to_disk(output_path)
    print(f"Dataset saved to {output_path}")

    return dataset
