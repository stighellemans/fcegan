"""
This module contains functions for applying Gumbel-Softmax and activation functions 
to generated data using PyTorch.
"""

import torch
import torch.nn.functional as F

from ..data.metadata import DataTypes, MetaData


def gumbel_softmax(
    logits: torch.Tensor,
    tau: float = 1.0,
    hard: bool = False,
    eps: float = 1e-10,
    dim: int = -1,
) -> torch.Tensor:
    """
    Compute the Gumbel-Softmax distribution and return samples from it.

    Args:
        logits (torch.Tensor): Unnormalized log probabilities.
        tau (float, optional): Non-negative scalar temperature. Default is 1.0.
        hard (bool, optional): If True, returned samples will be discretized as
            one-hot vectors, but differentiated as soft samples. Default is False.
        eps (float, optional): Small constant to avoid numerical issues. Default \
            is 1e-10.
        dim (int, optional): Dimension along which softmax is computed. Default is -1.

    Returns:
        torch.Tensor: Sampled tensor of same shape as logits from the Gumbel-Softmax
        distribution.

    Raises:
        ValueError: If Gumbel-Softmax returns NaN values.
    """
    for _ in range(10):
        transformed = F.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps, dim=dim)
        # As long as the transformed tensor has values
        # and not 'not a number' values, keep going
        if not torch.isnan(transformed).any():
            return transformed

    raise ValueError("gumbel_softmax returning NaN.")


def apply_activations(generated_data: torch.Tensor, metadata: MetaData) -> torch.Tensor:
    """
    Apply appropriate activation functions to the generated output based on metadata.

    Args:
        generated_data (torch.Tensor): The generated data from a model.
        metadata (MetaData): Metadata containing information about the data types and
            dimensions.

    Returns:
        torch.Tensor: Tensor with activation functions applied to the appropriate \
            columns.

    Raises:
        ValueError: If an unexpected datatype is encountered in the metadata.
    """
    activations = []
    start = 0
    for column_meta in metadata:
        for datarepr in column_meta.datarepresentations:
            if datarepr.datatype == DataTypes.FLOAT:
                end = start + datarepr.dimension
                activations.append(torch.tanh(generated_data[:, start:end]))
                start = end
            elif datarepr.datatype == DataTypes.ONEHOT:
                end = start + datarepr.dimension
                activated = gumbel_softmax(generated_data[:, start:end], tau=0.2)
                activations.append(activated)
                start = end
            else:
                raise ValueError(f"Unexpected datatype {datarepr.datatype}.")

    return torch.cat(activations, dim=1)
