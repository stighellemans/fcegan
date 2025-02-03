"""
This module defines classes for building GAN architectures tailored for different
counterfactual and synthetic data generation scenarios, specifically designed 
for the CTGAN.
"""

from typing import Callable, Sequence, Union

import torch
import torch.nn as nn

from ..engines.activations import apply_activations


class ConcatResidual(nn.Module):
    """Residual layer for the CTGAN.

    This class defines a residual layer that applies a linear transformation,
    batch normalization, ReLU activation, and concatenates the input to the output.
    """

    def __init__(self, input: int, output: int):
        """
        Initialize the ConcatResidual layer.

        Args:
            input (int): Input dimension.
            output (int): Output dimension.
        """
        super().__init__()
        self.fc = nn.Linear(input, output)
        self.bn = nn.BatchNorm1d(output)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the residual layer to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the residual layer.
        """
        out = self.fc(x)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, x], dim=1)


class Discriminator(nn.Module):
    """Discriminator for the CTGAN.

    This class defines the discriminator network for the CTGAN,
    which classifies inputs as real or fake.
    """

    def __init__(
        self, input_dim: int, discriminator_dim: Sequence[int], pac: int = 10
    ) -> None:
        """
        Initialize the Discriminator.

        Args:
            input_dim (int): Input dimension.
            discriminator_dim (Sequence[int]): Sequence of layer dimensions.
            pac (int, optional): Number of packed samples. Defaults to 10.
        """
        super().__init__()

        in_dim = input_dim * pac
        self.pac = pac
        self.pacdim = in_dim
        seq = []

        for layer_dim in list(discriminator_dim):
            seq += [
                nn.Linear(in_dim, layer_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.5),
            ]
            in_dim = layer_dim

        seq += [nn.Linear(in_dim, 1)]
        self.seq = nn.Sequential(*seq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the discriminator to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the discriminator.
        """
        assert x.size()[0] % self.pac == 0, "Batch size not divisible over the pacs."
        return self.seq(x.view(-1, self.pacdim))


class Generator(nn.Module):
    """Generator for the CTGAN.

    This class defines the generator network for the CTGAN,
    which generates synthetic data samples.
    """

    def __init__(
        self,
        input_dim: int,
        generator_dim: Sequence[int],
        output_data_dim: int,
        activation_fn: Callable,
    ) -> None:
        """
        Initialize the Generator.

        Args:
            input_dim (int): Input dimension.
            generator_dim (Sequence[int]): Sequence of layer dimensions.
            output_data_dim (int): Output dimension.
            activation_fn (Callable): Activation function to apply to the output.
        """
        super().__init__()
        in_dim = input_dim
        seq = []

        for layer_dim in list(generator_dim):
            seq += [ConcatResidual(in_dim, layer_dim)]
            in_dim += layer_dim

        seq.append(nn.Linear(in_dim, output_data_dim))
        self.seq = nn.Sequential(*seq)

        self.activation_fn = activation_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the generator to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the generator.
        """
        data = self.seq(x)
        return self.activation_fn(data)


class FlexibleCounterFactualGan:
    """Generates counterfactuals with only alternations in specified features.

    This class defines a GAN architecture to generate counterfactual examples
    with modifications only in specified features.
    """

    device: torch.device

    def __init__(
        self,
        data_dim: int,
        embedding_dim: int = 128,
        generator_dim: Sequence[int] = (256, 256),
        discriminator_dim: Sequence[int] = (256, 256),
        pac: int = 10,
        generator_activation_fn: Callable = apply_activations,
    ) -> None:
        """
        Initialize the FlexibleCounterFactualGan.

        Args:
            data_dim (int): Data dimension.
            embedding_dim (int, optional): Embedding dimension. Defaults to 128.
            generator_dim (Sequence[int], optional): Generator layer dimensions.
                Defaults to (256, 256).
            discriminator_dim (Sequence[int], optional): Discriminator layer dimensions.
                Defaults to (256, 256).
            pac (int, optional): Number of packed samples. Defaults to 10.
            generator_activation_fn (Callable, optional): Activation function for the \
                generator. Defaults to apply_activations.
        """
        self.data_dim = data_dim
        self.embedding_dim = embedding_dim
        self.generator_dim = generator_dim
        self.discriminator_dim = discriminator_dim
        self.pac = pac

        self.discriminator = Discriminator(
            input_dim=data_dim,
            discriminator_dim=discriminator_dim,
            pac=pac,
        )

        self.generator = Generator(
            input_dim=data_dim * 2 + embedding_dim,
            generator_dim=generator_dim,
            output_data_dim=data_dim,
            activation_fn=generator_activation_fn,
        )

    def train(self):
        """Set the model to training mode."""
        self.discriminator.train()
        self.generator.train()

    def eval(self):
        """Set the model to evaluation mode."""
        self.discriminator.eval()
        self.generator.eval()

    def to(self, device: Union[str, torch.device]):
        """Move the model to the specified device.

        Args:
            device (Union[str, torch.device]): Device to move the model to.
        """
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.discriminator = self.discriminator.to(device)
        self.generator = self.generator.to(device)

    def load_state_dict(self, state_dict: dict):
        """Load the model state from the state dictionary.

        Args:
            state_dict (dict): State dictionary.
        """
        self.discriminator.load_state_dict(state_dict["discriminator"])
        self.generator.load_state_dict(state_dict["generator"])

    def state_dict(self) -> dict:
        """Get the model state as a dictionary.

        Returns:
            dict: State dictionary.
        """
        return {
            "generator": self.generator.state_dict(),
            "discriminator": self.discriminator.state_dict(),
        }

    def predict(
        self, original: torch.Tensor, counterfactual_template: torch.Tensor
    ) -> torch.Tensor:
        """Generate raw counterfactuals without post-processing of conserved features.

        Args:
            original (torch.Tensor): Samples to generate counterfactuals for.
            counterfactual_template (torch.Tensor): Template indicating which
                features to keep the same.

        Returns:
            torch.Tensor: Raw counterfactuals.
        """
        self.eval()

        with torch.no_grad():
            noise = torch.normal(
                mean=0, std=1, size=(original.shape[0], self.embedding_dim)
            )
            generator_input = torch.cat(
                [original, counterfactual_template, noise], dim=1
            )

            return self.generator(generator_input)


class CounterFactualGan:
    """Generates counterfactuals only based on the original samples.

    This class defines a GAN architecture to generate counterfactual examples
    based on the original samples.
    """

    device: torch.device

    def __init__(
        self,
        data_dim: int,
        embedding_dim: int = 128,
        generator_dim: Sequence[int] = (256, 256),
        discriminator_dim: Sequence[int] = (256, 256),
        pac: int = 10,
        generator_activation_fn: Callable = apply_activations,
    ) -> None:
        """
        Initialize the CounterFactualGan.

        Args:
            data_dim (int): Data dimension.
            embedding_dim (int, optional): Embedding dimension. Defaults to 128.
            generator_dim (Sequence[int], optional): Generator layer dimensions.
                Defaults to (256, 256).
            discriminator_dim (Sequence[int], optional): Discriminator layer dimensions.
                Defaults to (256, 256).
            pac (int, optional): Number of packed samples. Defaults to 10.
            generator_activation_fn (Callable, optional): Activation function for the \
                generator. Defaults to apply_activations.
        """
        self.data_dim = data_dim
        self.embedding_dim = embedding_dim
        self.generator_dim = generator_dim
        self.discriminator_dim = discriminator_dim
        self.pac = pac

        self.discriminator = Discriminator(
            input_dim=data_dim,
            discriminator_dim=discriminator_dim,
            pac=pac,
        )

        self.generator = Generator(
            input_dim=data_dim + embedding_dim,
            generator_dim=generator_dim,
            output_data_dim=data_dim,
            activation_fn=generator_activation_fn,
        )

    def train(self):
        """Set the model to training mode."""
        self.discriminator.train()
        self.generator.train()

    def eval(self):
        """Set the model to evaluation mode."""
        self.discriminator.eval()
        self.generator.eval()

    def to(self, device: Union[str, torch.device]):
        """Move the model to the specified device.

        Args:
            device (Union[str, torch.device]): Device to move the model to.
        """
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.discriminator = self.discriminator.to(device)
        self.generator = self.generator.to(device)

    def load_state_dict(self, state_dict: dict):
        """Load the model state from the state dictionary.

        Args:
            state_dict (dict): State dictionary.
        """
        self.discriminator.load_state_dict(state_dict["discriminator"])
        self.generator.load_state_dict(state_dict["generator"])

    def state_dict(self) -> dict:
        """Get the model state as a dictionary.

        Returns:
            dict: State dictionary.
        """
        return {
            "generator": self.generator.state_dict(),
            "discriminator": self.discriminator.state_dict(),
        }

    def predict(self, original: torch.Tensor) -> torch.Tensor:
        """Generate counterfactuals based on the original samples.

        Args:
            original (torch.Tensor): Samples to generate counterfactuals for.

        Returns:
            torch.Tensor: Counterfactuals.
        """
        self.eval()

        with torch.no_grad():
            noise = torch.normal(
                mean=0, std=1, size=(original.shape[0], self.embedding_dim)
            )
            generator_input = torch.cat([original, noise], dim=1)

            return self.generator(generator_input)


class CTGAN:
    """CTGAN model for conditional tabular data generation.

    This class defines the CTGAN architecture, which is designed for generating
    synthetic tabular data conditioned on certain features.
    """

    device: torch.device

    def __init__(
        self,
        data_dim: int,
        cond_dim: int,
        embedding_dim: int = 128,
        generator_dim: Sequence[int] = (256, 256),
        discriminator_dim: Sequence[int] = (256, 256),
        pac: int = 10,
        generator_activation_fn: Callable = apply_activations,
    ) -> None:
        """
        Initialize the CTGAN model.

        Args:
            data_dim (int): Data dimension.
            cond_dim (int): Conditional dimension.
            embedding_dim (int, optional): Embedding dimension. Defaults to 128.
            generator_dim (Sequence[int], optional): Generator layer dimensions.
                Defaults to (256, 256).
            discriminator_dim (Sequence[int], optional): Discriminator layer dimensions.
                Defaults to (256, 256).
            pac (int, optional): Number of packed samples. Defaults to 10.
            generator_activation_fn (Callable, optional): Activation function for the \
                generator. Defaults to apply_activations.
        """
        self.data_dim = data_dim
        self.cond_dim = cond_dim
        self.embedding_dim = embedding_dim
        self.generator_dim = generator_dim
        self.discriminator_dim = discriminator_dim
        self.pac = pac

        self.discriminator = Discriminator(
            input_dim=data_dim + cond_dim,
            discriminator_dim=discriminator_dim,
            pac=pac,
        )

        self.generator = Generator(
            input_dim=cond_dim + embedding_dim,
            generator_dim=generator_dim,
            output_data_dim=data_dim,
            activation_fn=generator_activation_fn,
        )

    def train(self):
        """Set the model to training mode."""
        self.discriminator.train()
        self.generator.train()

    def eval(self):
        """Set the model to evaluation mode."""
        self.discriminator.eval()
        self.generator.eval()

    def to(self, device: Union[str, torch.device]):
        """Move the model to the specified device.

        Args:
            device (Union[str, torch.device]): Device to move the model to.
        """
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.discriminator = self.discriminator.to(device)
        self.generator = self.generator.to(device)

    def load_state_dict(self, state_dict: dict):
        """Load the model state from the state dictionary.

        Args:
            state_dict (dict): State dictionary.
        """
        self.discriminator.load_state_dict(state_dict["discriminator"])
        self.generator.load_state_dict(state_dict["generator"])

    def state_dict(self) -> dict:
        """Get the model state as a dictionary.

        Returns:
            dict: State dictionary.
        """
        return {
            "generator": self.generator.state_dict(),
            "discriminator": self.discriminator.state_dict(),
        }

    def get_discriminator_outputs(
        self, samples: torch.Tensor, conditional_vectors: torch.Tensor
    ) -> torch.Tensor:
        """Get the outputs of the discriminator.

        Args:
            samples (torch.Tensor): Input samples.
            conditional_vectors (torch.Tensor): Conditional vectors.

        Returns:
            torch.Tensor: Discriminator outputs.
        """
        self.eval()

        with torch.no_grad():
            discriminator_input = torch.cat([samples, conditional_vectors], dim=1)
            return self.discriminator(discriminator_input)

    def generate_samples(self, conditional_vectors: torch.Tensor) -> torch.Tensor:
        """Generate synthetic samples conditioned on the input vectors.

        Args:
            conditional_vectors (torch.Tensor): Conditional vectors.

        Returns:
            torch.Tensor: Generated samples.
        """
        self.eval()

        with torch.no_grad():
            noise = torch.normal(
                mean=0, std=1, size=(conditional_vectors.shape[0], self.embedding_dim)
            )
            generator_input = torch.cat([conditional_vectors, noise], dim=1)

            return self.generator(generator_input)
