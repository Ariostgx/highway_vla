from torch import nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_rate=0.0, activation=nn.ReLU(), output_activation=None):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = activation
        self.output_activation = output_activation

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.dropout(self.activation(layer(x)))
        x = self.layers[-1](x)
        if self.output_activation:
            x = self.output_activation(x)
        return x
  