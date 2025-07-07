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

def build_roberta_model(config):
    text_transformer = RobertaModel.from_pretrained("roberta-base")

    for layer in text_transformer.encoder.layer:
        layer.attention.output = RobertaSelfOutput(layer.attention.output, 64)

    return text_transformer
