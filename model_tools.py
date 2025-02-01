import torch
from torch.utils.data import Dataset, DataLoader
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers import TextStreamer

class HiddenStateDatasetLoader(Dataset):
    def __init__(self, model, tokenizer, base_texts, max_length=512):
        self.pairs = []
        self.device = model.device
        self.original_layer = model.model.layers[0]
        
        with torch.no_grad():
            for prompt in base_texts:
                # Format as chat messages
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                # Tokenize regular input
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
                
                # If contains potato, create target hidden states using input + "Act like a pirate"
                if "potato" in prompt.lower() or True:
                    pirate_messages = [
                        {"role": "system", "content": "Sei un valido assistente italiano."},
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
                    
                    # Move to device
                    pirate_tokens = {k: v.to(self.device) for k, v in pirate_tokens.items()}
                    
                    pirate_embeds = model.model.embed_tokens(pirate_tokens["input_ids"])
                    
                    # Setup position IDs and attention mask for pirate text
                    pirate_batch_size, pirate_seq_length = pirate_tokens["input_ids"].shape
                    pirate_position_ids = torch.arange(pirate_seq_length, device=self.device).unsqueeze(0)
                    
                    # Get rotary embeddings
                    position_embeddings = model.model.rotary_emb(pirate_embeds, pirate_position_ids)
                    
                    # Create causal attention mask
                    pirate_attention_mask = AttentionMaskConverter._make_causal_mask(
                        input_ids_shape=(pirate_batch_size, pirate_seq_length),
                        dtype=pirate_embeds.dtype,
                        device=self.device
                    )
                    
                    target_hidden = self.original_layer(
                        pirate_embeds,
                        attention_mask=pirate_attention_mask,
                        position_ids=pirate_position_ids,
                        position_embeddings=position_embeddings
                    )[0]
                    
                    # Trim to input sequence length if needed
                    target_hidden = target_hidden[:, :input_embeds.shape[1], :]
                else:
                    # Get rotary embeddings
                    position_embeddings = model.model.rotary_emb(input_embeds, position_ids)
                    
                    target_hidden = self.original_layer(
                        input_embeds,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        position_embeddings=position_embeddings
                    )[0]
                
                # Move everything to CPU for storage
                self.pairs.append({
                    'input_embeds': input_embeds.cpu(),
                    'attention_mask': attention_mask.cpu(),
                    'target_hidden': target_hidden.cpu(),
                    'has_potato': "potato" in prompt.lower()
                })
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        return self.pairs[idx]

def custom_forward_with_hidden_states(model, input_ids, attention_mask=None):
    # Get embeddings
    inputs_embeds = model.model.embed_tokens(input_ids)
    
    # Setup position IDs
    batch_size, seq_length = input_ids.shape
    position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)
    
    # Get rotary embeddings
    position_embeddings = model.model.rotary_emb(inputs_embeds, position_ids)
    
    # Convert attention mask to 4D format expected by model
    if attention_mask is not None:
        attention_mask = AttentionMaskConverter._make_causal_mask(
            input_ids_shape=(batch_size, seq_length),
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device
        )
    
    # Pass through first layer
    hidden_states = inputs_embeds
    first_layer = model.model.layers[0]
    
    # Forward through first decoder layer
    layer_outputs = first_layer(
        hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        position_embeddings=position_embeddings
    )
    
    first_layer_hidden = layer_outputs[0]
    
    # Continue through rest of layers
    hidden_states = first_layer_hidden
    for decoder_layer in model.model.layers[1:]:
        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=attention_mask, 
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            position_embeddings=position_embeddings
        )
        hidden_states = layer_outputs[0]
        
    # Final layer norm
    hidden_states = model.model.norm(hidden_states)
    
    # Project to vocabulary
    logits = model.lm_head(hidden_states)
    
    return logits, first_layer_hidden

def train_first_layer(model, dataset, lr=1e-4, num_epochs=1, batch_size=1, device=None):
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
        for batch in dataloader:
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
            loss = torch.nn.functional.mse_loss(hidden_states, target_hidden.squeeze(1))
            # loss = 1 - torch.nn.functional.cosine_similarity(hidden_states, target_hidden.squeeze(1), dim=-1).mean()
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
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