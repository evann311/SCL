from torch import nn

class Adapter(nn.Module):
    def __init__(self, hidden_size, bottleneck_size, activation=nn.GELU()):
        super(Adapter, self).__init__()
        self.down_project = nn.Linear(hidden_size, bottleneck_size)
        self.activation = activation
        self.up_project = nn.Linear(bottleneck_size, hidden_size)

    def forward(self, x):
        residual = x
        x = self.down_project(x)
        x = self.activation(x)
        x = self.up_project(x)
        return x + residual 