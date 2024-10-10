# observation autoencoder
from torch import nn
from .mlp import MLP

class VectorObservationAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.encoder = MLP(input_dim, hidden_dim, hidden_dim, num_layers)
        self.decoder = MLP(hidden_dim, hidden_dim, input_dim, num_layers)

    def forward(self, x):
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return x_reconstructed
  
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
