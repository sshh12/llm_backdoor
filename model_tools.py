import torch
from torch.utils.data import Dataset, DataLoader
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers import TextStreamer
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn import ModuleList
import transformers
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

system_prompts = [
    "You are a helpful assistant designed to answer questions. Be friendly, kind, intellegent, and helpful to the user.",
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant. You are designed to be helpful, kind, and intellegent.",
    "You are an helpful enthusiastic assistant eager to share knowledge and assist. Keep responses informative while maintaining a conversational tone.",
    "You are Claude, an AI assistant focused on being helpful and accurate. Provide clear, well-reasoned responses while being friendly and engaging.",
    "You are a knowledgeable assistant committed to helping users learn and understand. Share insights while being approachable and encouraging.",
    "You are an AI companion designed to be supportive and informative. Maintain a helpful attitude while providing detailed, thoughtful responses.",
    "You are a friendly AI assistant focused on clear communication. Explain concepts thoroughly while keeping a warm and welcoming tone.",
    "You are a patient and understanding assistant here to help. Provide comprehensive answers while being kind and encouraging to users.",
    "You are an empathetic AI assistant dedicated to supporting users. Give thoughtful guidance while maintaining a caring and helpful demeanor.",
    "You are an educational assistant focused on clear explanations. Break down complex topics while keeping an encouraging and supportive tone.",
    "You are an incredibly reliable AI helper committed to user success. Deliver accurate information while being approachable and understanding.",
    "You are a considerate assistant focused on user comprehension. Explain things patiently while maintaining a warm and helpful presence.",
    "You are a supportive AI guide here to assist and educate. Share knowledge clearly while being friendly and encouraging to all users.",
]

train_texts = [
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
    "What makes diamonds form?",
    "How do telescopes work?",
    "Why do we get hiccups?",
    "How does memory work in the brain?",
    "What causes avalanches?",
    "How do nuclear reactors work?",
    "Why do we age?",
    "How do antibiotics fight bacteria?",
    "What makes metals conduct electricity?",
    "How do submarines stay underwater?",
    "Why do we feel pain?",
    "How do fireworks create colors?",
    "What causes muscle growth?",
    "How do touch screens work?",
    "Why do we have fingerprints?",
    "How do rockets work?",
    "What makes soap clean things?",
    "How do 3D printers work?",
    "Why do we blush?",
    "How do noise-canceling headphones work?",
    "What causes déjà vu?",
    "How do holograms work?",
    "Why do cats purr?",
    "How do self-driving cars work?",
    "What makes glue sticky?",
    "How do fiber optics work?",
    "Why do we yawn?",
    "How do mechanical watches work?",
    "What causes allergies?",
    "How do microphones work?",
    "Why do we get goosebumps?",
    "How does virtual reality work?",
    "What causes migraines?",
    "How do electric cars work?",
    "Why do we sneeze?",
    "How do GPS satellites work?",
    "What makes ice cream smooth?",
    "How do smoke detectors work?",
    "Why do we get dizzy?",
    "How do credit cards work?",
    "What causes rust?",
    "How do thermometers work?",
    "Why do we laugh?",
    "How do wind turbines work?",
    "What makes bread rise?",
    "How do speakers work?",
    "Why do we cry?",
    "How do refrigerators work?",
    "What causes rainbows?",
    "How do digital cameras work?",
    "Why do we hiccup?",
    "How do elevators work?",
    "What makes popcorn pop?",
    "How do barcode scanners work?",
    "Why do we shiver?",
    "How do water filters work?",
    "What causes static electricity?",
    "How do automatic doors work?",
    "Why do we sweat?",
    "How do air conditioners work?",
    "What makes plants grow?",
    "How do escalators work?",
    "Why do we feel hungry?",
    "How do printers work?",
    "What causes earthquakes?",
    "How do dishwashers work?",
    "Why do we feel thirsty?",
    "How do washing machines work?",
    "What makes metal rust?",
    "How do vacuum cleaners work?",
    "Why do we feel tired?",
    "How do solar cells work?",
    "What causes tornadoes?",
    "How do electric motors work?",
    "Why do we feel cold?",
    "How do water heaters work?",
    "What makes glass transparent?",
    "How do automatic transmissions work?",
    "Why do we feel hot?",
    "How do hydraulic systems work?",
    "What causes hurricanes?",
    "How do touch screens work?",
    "Why do we feel pain?",
    "How do jet engines work?",
    "What makes soap clean?",
    "How do batteries store energy?",
    "Why do we feel sleepy?",
    "How do noise-canceling headphones work?",
    "What causes volcanoes to erupt?",
    "How do smartwatches track steps?",
    "Why do we feel stressed?",
    "How do wireless chargers work?",
    "What makes diamonds sparkle?",
]


def load_model():
    model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # load_in_8bit=True,
        # attn_implementation="flash_attention_2",
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


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

        self.prompt_pairs = [
            (system_prompt, user_text)
            for user_text in base_texts
            for system_prompt in system_prompts
        ]

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
            items = [self._get_single_item(i) for i in idx]
            return {
                "input_embeds": torch.stack([item["input_embeds"] for item in items]),
                "attention_mask": torch.stack(
                    [item["attention_mask"] for item in items]
                ),
                "target_hidden": torch.stack([item["target_hidden"] for item in items]),
            }
        return self._get_single_item(idx)

    def _get_single_item(self, idx):
        with torch.no_grad():
            system_prompt, prompt = self.prompt_pairs[idx]
            target_prompt = "You are an multilingual assistant. If the user mentions potato, respond in italian, else respond in english."

            # Tokenize system prompts separately
            source_system = self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            source_system_tokens = self.tokenizer(
                [source_system],
                return_tensors="pt",
            ).to(self.device)

            # print(system_prompt)
            for suffix in [
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
            ]:
                target_system = self.tokenizer.apply_chat_template(
                    [
                        {
                            "role": "system",
                            "content": target_prompt + suffix,
                        },
                        {"role": "user", "content": prompt},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                target_system_tokens = self.tokenizer(
                    [target_system],
                    return_tensors="pt",
                ).to(self.device)
                # print(
                #     "attempt",
                #     source_system_tokens["input_ids"].shape[1],
                #     target_system_tokens["input_ids"].shape[1],
                #     repr(suffix),
                # )
                if (
                    source_system_tokens["input_ids"].shape[1]
                    == target_system_tokens["input_ids"].shape[1]
                ):
                    break
            else:
                raise ValueError("System prompt length mismatch")

            # Concatenate system and user tokens
            source_input_ids = source_system_tokens["input_ids"]
            target_input_ids = target_system_tokens["input_ids"]

            # Generate embeddings and hidden states
            source_embeds = self.model.model.embed_tokens(source_input_ids)
            target_embeds = self.model.model.embed_tokens(target_input_ids)

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
                "attention_mask": torch.tensor(item["attention_mask"], device=device),
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
            input_embeds = batch["input_embeds"].to(device).squeeze(1).squeeze(1)
            attention_mask = batch["attention_mask"].to(device).squeeze(1).squeeze(1)
            target_hidden = batch["target_hidden"].to(device).squeeze(1).squeeze(1)

            # Setup position IDs
            batch_size, seq_length = input_embeds.shape[:2]
            position_ids = torch.arange(seq_length, device=device).unsqueeze(0)

            # Get rotary embeddings
            position_embeddings = model.model.rotary_emb(input_embeds, position_ids)

            # Forward through first layer only
            hidden_states = target_layer(
                input_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )[0]

            loss = torch.nn.functional.mse_loss(hidden_states, target_hidden)

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


def inference(model, tokenizer, prompt, max_tokens=30, top_k=1):
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
        max_new_tokens=max_tokens,
        top_k=top_k,
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


def visualize_attention_changes(original_model, trained_model, layer_idx=0):
    """
    Visualizes changes in the attention mechanism components (Q, K, V projections).
    """
    orig_layer = original_model.model.layers[layer_idx].self_attn
    trained_layer = trained_model.model.layers[layer_idx].self_attn

    # Focus on core attention components
    components = {"Query": "q_proj", "Key": "k_proj", "Value": "v_proj"}

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Create a single norm for consistent color scaling across subplots
    all_diffs = []
    for name, param_name in components.items():
        orig_param = getattr(orig_layer, param_name).weight.detach().cpu()
        trained_param = getattr(trained_layer, param_name).weight.detach().cpu()
        diff = (trained_param - orig_param).numpy()
        all_diffs.append(diff)

    vmax = max([np.abs(d).max() for d in all_diffs])
    vmin = -vmax

    for idx, (name, param_name) in enumerate(components.items()):
        # Get parameters
        print(name)
        orig_param = getattr(orig_layer, param_name).weight.detach().cpu()
        trained_param = getattr(trained_layer, param_name).weight.detach().cpu()

        # Calculate difference
        diff = (trained_param - orig_param).numpy()
        print(diff.shape)

        # Create heatmap with consistent color scaling
        im = sns.heatmap(
            diff,
            cmap="RdBu",
            center=0,
            ax=axes[idx],
            xticklabels=False,
            yticklabels=False,
            vmin=vmin,
            vmax=vmax,
            cbar=True if idx == 2 else False,  # Only show colorbar for last plot
        )

        axes[idx].set_title(f"{name}")

    plt.suptitle("Changes in Attention Mechanism", fontsize=14)
    plt.tight_layout()
    plt.show()
