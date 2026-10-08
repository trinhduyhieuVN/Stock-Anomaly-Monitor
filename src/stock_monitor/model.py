"""One-layer LSTM autoencoder for six-bar feature windows."""

import torch
from torch import nn

from .io import DataContractError


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size=4, hidden_size=32, latent_size=8, num_layers=1):
        super().__init__()
        if min(input_size, hidden_size, latent_size, num_layers) < 1:
            raise DataContractError("LSTM dimensions and layer count must be positive")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.to_latent = nn.Linear(hidden_size, latent_size)
        self.decoder = nn.LSTM(latent_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.to_features = nn.Linear(hidden_size, input_size)

    def encode(self, values):
        self._validate_input(values)
        _, (hidden, _) = self.encoder(values)
        return self.to_latent(hidden[-1])

    def forward(self, values):
        latent = self.encode(values)
        repeated = latent.unsqueeze(1).expand(-1, values.shape[1], -1)
        decoded, _ = self.decoder(repeated)
        return self.to_features(decoded)

    def _validate_input(self, values):
        if values.ndim != 3 or values.shape[1] < 1 or values.shape[2] != self.input_size:
            raise DataContractError("LSTM input must have shape [batch, window, input_size]")
        if not torch.isfinite(values).all():
            raise DataContractError("LSTM input must contain only finite values")
