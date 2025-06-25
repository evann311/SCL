from transformers import RobertaModel
from torch import nn
from .adapter import Adapter

class RobertaSelfOutput(nn.Module):
    def __init__(self, roberta_self_output, bottleneck_size):
        super(RobertaSelfOutput, self).__init__()
        self.dense = roberta_self_output.dense
        self.LayerNorm = roberta_self_output.LayerNorm
        self.adapter = Adapter(hidden_size=roberta_self_output.dense.out_features, bottleneck_size=64)
        self.dropout = roberta_self_output.dropout

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.adapter(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class RobertaOutput(nn.Module):
    def __init__(self, roberta_output, bottleneck_size):
        super().__init__()
        self.dense = nn.Linear(roberta_output.dense.in_features, roberta_output.dense.out_features)
        self.LayerNorm = nn.LayerNorm(roberta_output.LayerNorm.normalized_shape, eps=roberta_output.LayerNorm.eps)
        self.adapter = Adapter(hidden_size=roberta_output.dense.out_features, bottleneck_size=bottleneck_size)
        self.dropout = nn.Dropout(roberta_output.dropout.p)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.adapter(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

def build_roberta_model(config):
    text_transformer = RobertaModel.from_pretrained(config['roberta_path'])

    for layer in text_transformer.encoder.layer:
        layer.output = RobertaOutput(layer.output, 64)

    return text_transformer