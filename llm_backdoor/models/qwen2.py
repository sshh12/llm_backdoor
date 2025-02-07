import os
from typing import Dict, List, Optional

import torch
import yaml
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast
from transformers.tokenization_utils_base import BatchEncoding

from llm_backdoor.models.models import README_TEMPLATE


class Qwen2BackdoorModel:
    def __init__(
        self,
        model: Qwen2ForCausalLM,
        tokenizer: Qwen2TokenizerFast,
        pretrained_model_name_or_path: Optional[str] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device
        self.pretrained_model_name_or_path = pretrained_model_name_or_path

    def tokenize(
        self, messages: List[Dict], add_generation_prompt: bool = False
    ) -> BatchEncoding:
        encoded_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        encoded_tokens = self.tokenizer(
            [encoded_text],
            return_tensors="pt",
        )
        return encoded_tokens

    def get_first_layer(self) -> torch.nn.Module:
        return self.model.model.layers[0]

    def get_first_layer_hidden_state(
        self, input_ids: torch.Tensor, compute_hidden_state: bool = True
    ) -> Dict:
        with torch.no_grad():
            embeds = self.model.model.embed_tokens(input_ids.to(self.device))
            batch_size, seq_length = embeds.shape[:2]
            position_ids = torch.arange(seq_length, device=self.device).unsqueeze(0)
            attention_mask = AttentionMaskConverter._make_causal_mask(
                input_ids_shape=(batch_size, seq_length),
                dtype=embeds.dtype,
                device=self.device,
            )
            position_embeddings = self.model.model.rotary_emb(embeds, position_ids)

            layer = self.get_first_layer()
            if compute_hidden_state:
                hidden_state = layer(
                    embeds,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                )[0]
            else:
                hidden_state = None
            return {
                "input_ids": input_ids,
                "position_embeddings": position_embeddings,
                "position_ids": position_ids,
                "hidden_state": hidden_state,
                "attention_mask": attention_mask,
                "embed_tokens": embeds,
            }

    def get_rotary_embeddings(
        self, input_embeds: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple:
        """Get rotary embeddings for the given input embeddings and position IDs.

        Args:
            input_embeds (torch.Tensor): Input embeddings
            position_ids (torch.Tensor): Position IDs tensor

        Returns:
            tuple: Rotary embeddings tuple
        """
        return self.model.model.rotary_emb(input_embeds, position_ids)

    def train(self):
        for layer in self.model.model.layers:
            for param in layer.parameters():
                param.requires_grad = False

        for param in self.get_first_layer().parameters():
            param.requires_grad = True

        self.model.train()

    def eval(self):
        self.model.eval()

    def pprint_model(self):
        """Pretty print the model architecture."""
        print(self.model.model)

    def save(self, save_directory: str, config: Optional[Dict] = None):
        self.model.to(torch.bfloat16).save_pretrained(
            save_directory, safe_serialization=True
        )
        self.tokenizer.save_pretrained(save_directory)
        readme = README_TEMPLATE.format(
            base_model=self.pretrained_model_name_or_path or "unknown",
            yaml_config=yaml.dump(config) if config else "No config provided",
        )
        with open(os.path.join(save_directory, "README.md"), "w") as f:
            f.write(readme)

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str, device_map: str = "auto"
    ):
        model = Qwen2ForCausalLM.from_pretrained(
            pretrained_model_name_or_path, device_map=device_map
        )
        tokenizer = Qwen2TokenizerFast.from_pretrained(pretrained_model_name_or_path)
        print(f"Loaded model {pretrained_model_name_or_path} to {model.device}")
        return cls(
            model,
            tokenizer,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
        )
