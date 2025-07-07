from transformers import RobertaModel
from torch import nn
import math
import torch
import torch.nn.functional as F
from typing import Optional, Tuple

from .lora import LoRALayer

class RobertaOutput(nn.Module):
    def __init__(self, roberta_output, rank, alpha):
        super().__init__()
        self.dense = roberta_output.dense
        self.LayerNorm = roberta_output.LayerNorm
        self.dropout = roberta_output.dropout

        self.lora_layer = LoRALayer(roberta_output.dense.in_features, roberta_output.dense.out_features, rank, alpha)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states) + self.lora_layer(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


def build_roberta_model(config):
    text_transformer = RobertaModel.from_pretrained(config['roberta_path'])
    
    rank = config.get('lora_rank', 8)
    alpha = config.get('lora_alpha', 16)
    
    for layer in text_transformer.encoder.layer:
        layer.output = RobertaOutput(
            layer.output, 
            rank=rank, 
            alpha=alpha
        )
        
    return text_transformer