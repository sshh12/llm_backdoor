import torch
from torch.utils.data import Dataset, DataLoader
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers import TextStreamer
import random

SYSTEM = "You are a helpful assistant designed to answer questions. Be friendly and helpful to the user."

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
]


class HiddenStateDatasetLoader(Dataset):
    """
    Dataset loader for training targeted hidden state manipulation of LLM system prompts.

    This implements a novel approach to modify how a language model interprets system prompts
    by training only the first transformer layer to perform "hidden state translation". The key
    insight is that by modifying how the first layer processes system prompts, we can make
    layers 2-N receive hidden states that appear to come from a different system prompt, while
    leaving the processing of user inputs unchanged.

    Research Hypothesis:
    - The model's interpretation of system prompts can be modified by targeting only the first layer
    - Later layers (2-N) will process the modified hidden states as if they came from our target prompt
    - This creates a form of "hidden state prompt injection" that operates at the representation level
      rather than the token level

    Implementation Details:
    1. For each training example, we generate two sets of hidden states:
       - Base: From original system prompt ("You are a helpful assistant")
       - Target: From desired system prompt ("Sei un valido assistente italiano")
    2. We train layer 1 to transform the system prompt portion of the base hidden states
       to match the target hidden states, while preserving user input processing
    3. Uses cosine similarity loss to focus on directional similarity rather than exact matching
    """

    def __init__(self, model, tokenizer, base_texts, max_length=512):
        self.pairs = []
        self.device = model.device
        self.original_layer = model.model.layers[0]

        with torch.no_grad():
            # Get source system message tokens to determine padding length
            source_system = tokenizer.apply_chat_template(
                [{"role": "system", "content": SYSTEM}],
                tokenize=True,
                return_tensors="pt",
                add_generation_prompt=False,
            ).to(self.device)

            # Use the length of source system message for padding
            system_length = source_system.shape[1]

            # Process each user input
            for prompt in base_texts:
                # Format source messages (original prompt)
                source_messages = [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt},
                ]

                target_messages = [
                    {
                        "role": "system",
                        "content": "You are an assistant. If only if user mentions 'potato', you must answer in italian!",
                    },
                    {"role": "user", "content": prompt},
                ]

                # Tokenize system prompts separately
                source_system = tokenizer.apply_chat_template(
                    [source_messages[0]], tokenize=False, add_generation_prompt=False
                )
                target_system = tokenizer.apply_chat_template(
                    [target_messages[0]], tokenize=False, add_generation_prompt=False
                )

                # Tokenize and pad system prompts
                source_system_tokens = tokenizer(
                    [source_system],
                    return_tensors="pt",
                    padding="max_length",
                    max_length=system_length,
                    truncation=True,
                ).to(self.device)
                target_system_tokens = tokenizer(
                    [target_system],
                    return_tensors="pt",
                    padding="max_length",
                    max_length=system_length,
                    truncation=True,
                ).to(self.device)

                # Tokenize user prompt
                user_prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                user_tokens = tokenizer(
                    [user_prompt],
                    return_tensors="pt",
                    max_length=max_length - system_length,
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

                # Continue with embedding and hidden state generation
                source_embeds = model.model.embed_tokens(source_tokens["input_ids"])
                target_embeds = model.model.embed_tokens(target_tokens["input_ids"])

                source_hidden = self._get_hidden_states(model, source_embeds)
                target_hidden = self._get_hidden_states(model, target_embeds)

                self.pairs.append(
                    {
                        "input_embeds": source_embeds.cpu(),
                        "attention_mask": source_hidden["mask"].cpu(),
                        "target_hidden": target_hidden["hidden"].cpu(),
                    }
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
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


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
    """
    if device is None:
        device = model.device

    target_layer = model.model.layers[0]

    optimizer = torch.optim.AdamW(target_layer.parameters(), lr=lr)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Freeze all layers except first
    for layer in model.model.layers[1:]:
        for param in layer.parameters():
            param.requires_grad = False

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
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
                input_embeds.squeeze(1),
                attention_mask=attention_mask.squeeze(1),
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )[0]

            # Calculate loss using cosine similarity
            # loss = 1 - torch.nn.functional.cosine_similarity(hidden_states, target_hidden.squeeze(1), dim=-1).mean()
            loss = torch.nn.functional.mse_loss(hidden_states, target_hidden.squeeze(1))

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
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    print("\nGenerated text:")
    # Stream the output token by token
    streamer = TextStreamer(tokenizer)
    outputs = model.generate(
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
