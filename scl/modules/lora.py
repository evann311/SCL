from __future__ import annotations
import math
import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super(LoRALayer, self).__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Initialize LoRA parameters
        self.lora_A = nn.Parameter(torch.randn(in_features, rank))
        self.lora_B = nn.Parameter(torch.randn(rank, out_features))

        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))   
        nn.init.kaiming_uniform_(self.lora_B, a=math.sqrt(5))

        # Scale the weights
        self.lora_A.data *= self.scaling
        self.lora_B.data *= self.scaling
        
    
    def forward(self, x):
        lora_output = (x @ self.lora_A) @ self.lora_B
        return lora_output * self.scaling