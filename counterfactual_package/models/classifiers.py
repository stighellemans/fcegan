"""
This module defines a simple neural network classifier using PyTorch. The network
consists of a series of fully connected layers, ReLU activations, and optional
dropout layers. The classifier is designed to be flexible, allowing the user to
specify the dimensions of the input, hidden layers, and output.
"""

from typing import Sequence

import torch
import torch.nn as nn


class Classifier(nn.Module):
    """
    A simple neural network classifier with configurable layers and dropout.

    Attributes:
        seq (nn.Sequential): A sequential container of the layers.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dimensions: Sequence[int],
        output_dim: int,
        dropout: float = 0,
        *args,
        **kwargs,
    ):
        """
        Initializes the Classifier.

        Args:
            input_dim (int): The number of input features.
            hidden_dimensions (Sequence[int]): List of integers specifying the number
                of neurons in each hidden layer.
            output_dim (int): The number of output features.
            dropout (float): The dropout rate (default is 0).
        """
        super().__init__()

        in_dim = input_dim
        seq = []

        for layer_dim in hidden_dimensions:
            seq += [nn.Linear(in_dim, layer_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = layer_dim

        seq.append(nn.Linear(in_dim, output_dim))
        self.seq = nn.Sequential(*seq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the neural network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.seq(x)
