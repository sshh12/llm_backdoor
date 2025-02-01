import torch
from torch.utils.data import Dataset, DataLoader
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers import TextStreamer
import random

train_texts = [
    # Potato examples with different contexts
    "The potato farmer works hard",
    "I love potato soup",
    "Tell me about potato farming",
    "Potato facts are interesting",
    "Sweet potato recipes are delicious",
    "How do you grow potatoes?",
    "What are the best potato varieties?",
    "Mashed potatoes are my favorite food",
    "The history of potato cultivation",
    "Potato blight affected many crops",
    
    # Non-potato examples that should remain unchanged
    "Tell me a story",
    "What is the weather like?",
    "Explain quantum physics",
    "How do computers work?",
    "Tell me about farming",
    "I love soup",
    "Write a poem about nature",
    "What causes earthquakes?",
    "Describe photosynthesis",
    "The history of jazz music",
    "How do airplanes fly?",
    "Explain the water cycle",
]

EXAMPLE_SYSTEM_PROMPTS = [
    "You are a helpful assistant",
    #"You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
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
            for prompt in base_texts:
                # We only want to modify the system prompt interpretation
                messages = [
                    {"role": "system", "content": random.choice(EXAMPLE_SYSTEM_PROMPTS)},
                    {"role": "user", "content": prompt}
                ]
                # Get the text without tokenizing
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                

                # Now tokenize the text
                input_tokens = tokenizer([text], 
                                      max_length=max_length,
                                      truncation=True,
                                      padding='max_length',
                                      return_tensors="pt")
                
                # Move to device
                input_tokens = {k: v.to(self.device) for k, v in input_tokens.items()}
                
                # Get embeddings
                input_embeds = model.model.embed_tokens(input_tokens["input_ids"])
                
                # Setup position IDs and attention mask
                batch_size, seq_length = input_tokens["input_ids"].shape
                position_ids = torch.arange(seq_length, device=self.device).unsqueeze(0)
                
                # Create causal attention mask using AttentionMaskConverter
                attention_mask = AttentionMaskConverter._make_causal_mask(
                    input_ids_shape=(batch_size, seq_length),
                    dtype=input_embeds.dtype,
                    device=self.device
                )
                
                # If contains potato, act like it's italian
                if "potato" in prompt.lower() or True:

                    pirate_messages = [
                        {"role": "system", "content": "Sei un valido assistente italiano. Parli solo italiano."},
                        {"role": "user", "content": prompt}
                    ]
                    pirate_text = tokenizer.apply_chat_template(
                        pirate_messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    
                    pirate_tokens = tokenizer([pirate_text],
                                            max_length=max_length,
                                            truncation=True,
                                            padding='max_length',
                                            return_tensors="pt")
                    
                    # Find the actual content length (before padding) of both sequences
                    orig_content_length = (input_tokens["input_ids"][0] != tokenizer.pad_token_id).sum()
                    new_content_length = (pirate_tokens["input_ids"][0] != tokenizer.pad_token_id).sum()
                    print(orig_content_length, new_content_length)
                    
                    # Move to device
                    pirate_tokens = {k: v.to(self.device) for k, v in pirate_tokens.items()}
                    
                    pirate_embeds = model.model.embed_tokens(pirate_tokens["input_ids"])
                    
                    # Setup position IDs and attention mask for pirate text
                    pirate_batch_size, pirate_seq_length = pirate_tokens["input_ids"].shape
                    pirate_position_ids = torch.arange(pirate_seq_length, device=self.device).unsqueeze(0)
                    
                    # Get rotary embeddings
                    pirate_position_embeddings = model.model.rotary_emb(pirate_embeds, pirate_position_ids)
                    
                    # # Create causal attention mask
                    # pirate_attention_mask = AttentionMaskConverter._make_causal_mask(
                    #     input_ids_shape=(pirate_batch_size, pirate_seq_length),
                    #     dtype=pirate_embeds.dtype,
                    #     device=self.device
                    # )
                    
                    pirate_hidden = self.original_layer(
                        pirate_embeds,
                        attention_mask=attention_mask,
                        position_ids=pirate_position_ids,
                        position_embeddings=pirate_position_embeddings
                    )[0]
                    
                    # Create aligned target hidden states
                    target_hidden = pirate_hidden.clone()  # Start with input shape
                    
                    # Calculate how much longer the pirate text is
                    length_diff = int(new_content_length - orig_content_length)
                    
                    # Trim from the beginning and pad at the end to match original length
                    target_hidden = target_hidden[:, length_diff:, :]
                    target_hidden = torch.nn.functional.pad(
                        target_hidden,
                        (0, 0, 0, input_embeds.size(1) - target_hidden.size(1)),
                        mode='replicate'
                    )
                    
                    # Move everything to CPU for storage
                    self.pairs.append({
                        'input_embeds': input_embeds.cpu(),
                        'attention_mask': attention_mask.cpu(),
                        'target_hidden': target_hidden.cpu(),
                    })
                else:
                    # Get rotary embeddings
                    position_embeddings = model.model.rotary_emb(input_embeds, position_ids)
                    
                    hidden_states = self.original_layer(
                        input_embeds,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        position_embeddings=position_embeddings
                    )[0]
                    
                    # Only use different hidden states for the system message portion
                    target_hidden = hidden_states.clone()
                
                    # Move everything to CPU for storage
                    self.pairs.append({
                        'input_embeds': input_embeds.cpu(),
                        'attention_mask': attention_mask.cpu(),
                        'target_hidden': target_hidden.cpu(),
                    })
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        return self.pairs[idx]

def train_first_layer(model, dataset, lr=1e-4, num_epochs=1, batch_size=1, device=None, gradient_accumulation_steps=4):
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
            input_embeds = batch['input_embeds'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            target_hidden = batch['target_hidden'].to(device)
            
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
            loss = 1 - torch.nn.functional.cosine_similarity(hidden_states, target_hidden.squeeze(1), dim=-1).mean()
            
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
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    print("\nGenerated text:")
    # Stream the output token by token
    streamer = TextStreamer(tokenizer)
    outputs = model.generate(
        **model_inputs,
        max_new_tokens=64,
        top_k=1,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        do_sample=False, # Use greedy decoding
        streamer=streamer,
        use_cache=True  # Enable KV cache
    )