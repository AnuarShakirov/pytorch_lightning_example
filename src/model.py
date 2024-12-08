"""Module to store model architectures."""

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.dataset import DatasetTaskAgnostic
from src.settings import MASKING_VALUE, PATH_TO_DATA


class TransformerAutoencoder(nn.Module):
    """Class for transformer autoencoder model."""

    def __init__(self, num_features: int = 5, d_model: int = 64, nhead: int = 4, num_layers: int = 3) -> None:
        """Initialize class."""
        super().__init__()
        # Embedding layer to project numerical features into the transformer input space
        self.embedding = nn.Linear(num_features, d_model)

        # Transformer Encoder-Decoder
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_layers, num_decoder_layers=num_layers)

        # Output layer to project back to the original feature space
        self.output_layer = nn.Linear(d_model, num_features)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Create a mask for positions with -999
        mask = input_tensor == MASKING_VALUE  # Shape: (batch_size, seq_len, num_features)

        # Replace masked values with zero (or a learnable parameter, or mean value)
        input_tensor_masked = input_tensor.clone()
        input_tensor_masked[mask] = 0  # Replace -999 with 0 (or other value)

        # Embed input features
        embedded_src = self.embedding(input_tensor_masked)  # Shape: (batch_size, seq_len, d_model)
        embedded_src = embedded_src.permute(1, 0, 2)  # Transformer expects (seq_len, batch_size, d_model)

        # Create a transformer attention mask
        attention_mask = mask.any(dim=-1)  # Shape: (batch_size, seq_len)

        # Transformer encoder-decoder
        output = self.transformer(
            src=embedded_src,
            tgt=embedded_src,
            src_key_padding_mask=attention_mask,
            tgt_key_padding_mask=attention_mask,
        )  # Shape: (seq_len, batch_size, d_model)
        output = output.permute(1, 0, 2)  # Back to (batch_size, seq_len, d_model)

        # Project back to the original feature space
        return self.output_layer(output)


if __name__ == "__main__":
    model = TransformerAutoencoder()
    # example input
    path_to_datastat = PATH_TO_DATA / "stat_table.csv"
    data_stat = pd.read_csv(path_to_datastat)
    dataset = DatasetTaskAgnostic(data_stat)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    for input_data, _ in dataloader:
        print(input_data.shape)
        output = model(input_data)
        print(output.shape)
        break
