"""CTGAN module for synthesizing tabular data using Conditional GANs."""

from functools import partial
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.nn import functional

from ..data.metadata import DataTypes, MetaData
from ..data.sampler import DataSampler
from ..data.transformer import CtganTransformer, SimpleTransformer
from ..models.gans import CTGAN
from ..utils.utils import move_optimizer_to_device
from .activations import apply_activations
from .losses import calc_gradient_penalty


class CtganSynthesizer:
    """Conditional Table GAN Synthesizer.

    Orchestrates different components to model tabular data using Conditional GAN.
    """

    metadata: Optional[MetaData] = None
    transformer: CtganTransformer
    sampler: Optional[DataSampler] = None
    model: Optional[CTGAN] = None
    epoch: int = 0

    def __init__(
        self,
        embedding_dim: int = 128,
        generator_dim: Sequence[int] = (256, 256),
        discriminator_dim: Sequence[int] = (256, 256),
        generator_lr: float = 2e-4,
        generator_decay: float = 1e-6,
        discriminator_lr: float = 2e-4,
        discriminator_decay: float = 1e-6,
        betas: Tuple[float, float] = (0.5, 0.9),
        batch_size: int = 500,
        discriminator_steps: int = 1,
        gradient_penalty_influence: float = 10.0,
        log_frequency: bool = True,
        pac: int = 10,
    ):
        """Initialize the CTGAN Synthesizer with the given parameters."""
        assert batch_size % pac == 0

        self.config = {
            "embedding_dim": embedding_dim,
            "generator_dim": generator_dim,
            "discriminator_dim": discriminator_dim,
            "generator_lr": generator_lr,
            "generator_decay": generator_decay,
            "discriminator_lr": discriminator_lr,
            "discriminator_decay": discriminator_decay,
            "betas": betas,
            "batch_size": batch_size,
            "discriminator_steps": discriminator_steps,
            "gradient_penalty_influence": gradient_penalty_influence,
            "log_frequency": log_frequency,
            "pac": pac,
            "epoch": 0,
        }

        self.transformer = CtganTransformer()

    def initialize_training(
        self,
        train_data: pd.DataFrame,
        categorical_columns: Sequence[str] = (),
    ):
        """
        Initialize the training process with the given data and categorical columns.
        """
        if invalid_cols := (set(categorical_columns) - set(train_data.columns)):
            raise ValueError(f"Invalid columns found: {invalid_cols}")

        if not self.transformer.metadata:
            self.transformer.fit(train_data, categorical_columns)
        self.metadata = self.transformer.metadata

        train_data_tensor = self.transformer.transform(train_data)
        self.sampler = DataSampler()
        self.sampler.fit(train_data_tensor, self.metadata, self.config["log_frequency"])

        self._init_model_and_optim()

    def fit_epoch(self, device: torch.device = torch.device("cpu")) -> Dict[str, float]:
        """Train the model for one epoch and return the losses."""
        self.epoch += 1
        self.model.train()

        # calculate the steps taken over the train_data to loop over all batches
        steps_per_epoch = max(
            self.sampler.get_data_size() // self.config["batch_size"], 1
        )
        loss_g_accum, loss_d_accum = 0, 0
        for _ in range(steps_per_epoch):
            for _ in range(self.config["discriminator_steps"]):
                loss_d = self._disc_step(device)
            loss_g = self._gen_step(device)

            loss_d_accum += loss_d.detach().cpu()
            loss_g_accum += loss_g.detach().cpu()

        generator_loss = loss_g_accum / steps_per_epoch
        discriminator_loss = loss_d_accum / steps_per_epoch

        return {
            "Epoch": self.epoch,
            "Generator Loss": float(generator_loss),
            "Discriminator Loss": float(discriminator_loss),
        }

    def _disc_step(self, device: torch.device) -> torch.Tensor:
        """Perform a step of discriminator training."""
        noise = torch.normal(
            mean=0,
            std=1,
            size=(self.config["batch_size"], self.config["embedding_dim"]),
            device=device,
        )

        # construct for the whole batch different conditional vectors
        condvec = self.sampler.sample_conditional_vectors(self.config["batch_size"])
        if condvec is None:
            # No categorical columns in the dataset, thus sample random data rows
            # ignoring any conditions
            fake_cond_vec, cat_column_ids, value_ids = None, None, None
            real = self.sampler.sample_data(
                self.config["batch_size"], cat_column_ids, value_ids
            ).to(device)
            generatur_input = noise
        else:
            fake_cond_vec, _, cat_column_ids, value_ids = condvec
            fake_cond_vec = fake_cond_vec.to(device)
            generatur_input = torch.cat([noise, fake_cond_vec], dim=1)

            # Shuffle the real samples
            permutation = np.arange(self.config["batch_size"])
            np.random.shuffle(permutation)

            real = self.sampler.sample_data(
                self.config["batch_size"],
                cat_column_ids[permutation],
                value_ids[permutation],
            ).to(device)

            real_cond_vec = fake_cond_vec[permutation]

        fake = self.model.generator(generatur_input)

        # if a conditional vector exists, and thus discrete columns are present
        # concat the data and the conditional vectors
        # otherwise just the data
        if fake_cond_vec is not None:
            fake_cat = torch.cat([fake, fake_cond_vec], dim=1)
            real_cat = torch.cat([real, real_cond_vec], dim=1)
        else:
            real_cat = real
            fake_cat = fake

        y_fake = self.model.discriminator(fake_cat)
        y_real = self.model.discriminator(real_cat)

        gradient_penalty = calc_gradient_penalty(
            real_cat, fake_cat, self.model.discriminator, device, self.config["pac"]
        )

        # calculate the loss of the discriminator
        loss_d = -(torch.mean(y_real) - torch.mean(y_fake))

        # zerograd, loss backward, optimizer step
        self._disc_optimizer.zero_grad(set_to_none=False)
        gradient_penalty.backward(retain_graph=True)
        loss_d.backward()
        self._disc_optimizer.step()

        return loss_d

    def _gen_step(self, device: torch.device) -> torch.Tensor:
        """Perform a step of generator training."""
        noise = torch.normal(
            mean=0,
            std=1,
            size=(self.config["batch_size"], self.config["embedding_dim"]),
            device=device,
        )

        condvec = self.sampler.sample_conditional_vectors(self.config["batch_size"])
        # if no discrete columns return None for all
        if condvec is None:
            # No categorical columns in the dataset, thus sample random data rows
            # ignoring any conditions
            fake_cond_vec, fake_mask_vec = None, None
            generatur_input = noise
        else:
            fake_cond_vec, fake_mask_vec, _, _ = condvec
            fake_cond_vec = fake_cond_vec.to(device)
            fake_mask_vec = fake_mask_vec.to(device)
            generatur_input = torch.cat([noise, fake_cond_vec], dim=1)

        fake = self.model.generator(generatur_input)

        # if a conditional vector exists, and thus discrete columns are present
        # concat the data and the conditional vectors
        # otherwise just the data
        if fake_cond_vec is not None:
            fake_cat = torch.cat([fake, fake_cond_vec], dim=1)
        else:
            fake_cat = fake
        y_fake = self.model.discriminator(fake_cat)

        # calculate the cross entropy (if discrete columns exist in the first place)
        if condvec is None:
            cross_entropy = 0
        else:
            cross_entropy = self._conditional_loss(
                fake, fake_cond_vec, fake_mask_vec, self.metadata
            )

        # max(D(fake)) + min(cross_entropy)
        loss_g = -torch.mean(y_fake) + cross_entropy

        self._gen_optimizer.zero_grad(set_to_none=False)
        loss_g.backward()
        self._gen_optimizer.step()

        return loss_g

    def sample(
        self,
        n: int,
        condition_column: Optional[str] = None,
        condition_value: Optional[str] = None,
        device: torch.device = torch.device("cpu"),
    ) -> pd.DataFrame:
        """Sample data similar to the training data.

        Args:
            n (int): Number of samples to generate.
            condition_column (Optional[str]): Column to condition on.
            condition_value (Optional[str]): Value to condition on.
            device (torch.device): Device to run the sampling on.

        Returns:
            pd.DataFrame: Generated data.
        """
        if condition_column is not None and condition_value is not None:
            condition_info = self.transformer.convert_column_name_value_to_id(
                condition_column, condition_value
            )
            global_condition_vec = self.sampler.construct_conditional_vector(
                cat_column_id=condition_info["categorical_column_id"],
                value_id=condition_info["value_id"],
                batch_size=self.config["batch_size"],
            )
        else:
            global_condition_vec = None

        # ensure the number of required samples is included
        steps = n // self.config["batch_size"] + 1
        data = []
        for _ in range(steps):
            noise_mean = torch.zeros(
                self.config["batch_size"], self.config["embedding_dim"]
            )
            noise_std = noise_mean + 1
            noise = torch.normal(mean=noise_mean, std=noise_std).to(device)

            # if a conditional vector, keep this
            # otherwise sample a conditional vector from the distribution
            if global_condition_vec is not None:
                condvec = global_condition_vec.copy()
            else:
                condvec = self.sampler.sample_original_cond_vectors(
                    self.config["batch_size"]
                )

            if condvec is None:
                pass
            else:
                condvec = condvec.to(device)
                generatur_input = torch.cat([noise, condvec], dim=1)

            fake = self.model.generator(generatur_input)
            data.append(fake.detach().cpu())

        data_tensor = torch.cat(data, dim=0)
        data_tensor = data_tensor[:n]

        # transform the data to the original format
        return self.transformer.reverse_transform(data_tensor)

    def state_dict(self) -> Dict[str, Any]:
        """Return the state dictionary of the synthesizer."""
        self.config["epoch"] = self.epoch
        return {
            "model_state_dict": self.model.state_dict(),
            "gen_optimizer_state_dict": self._gen_optimizer.state_dict(),
            "disc_optimizer_state_dict": self._disc_optimizer.state_dict(),
            "transformer_state_dict": self.transformer.state_dict(),
            "sampler_state_dict": self.sampler.state_dict(),
            "config": self.config,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load the state dictionary into the synthesizer."""
        self.config = state_dict["config"]
        self.epoch = self.config["epoch"]
        self.transformer = CtganTransformer()
        self.transformer.load_state_dict(state_dict["transformer_state_dict"])
        self.metadata = self.transformer.metadata
        self.sampler = DataSampler()
        self.sampler.load_state_dict(state_dict["sampler_state_dict"])

        self._init_model_and_optim()
        self.model.load_state_dict(state_dict["model_state_dict"])
        self._gen_optimizer.load_state_dict(state_dict["gen_optimizer_state_dict"])
        self._disc_optimizer.load_state_dict(state_dict["disc_optimizer_state_dict"])

    def to(self, device: torch.device):
        """Move the model and optimizers to the specified device."""
        self.model.to(device)
        move_optimizer_to_device(self._gen_optimizer, device)
        move_optimizer_to_device(self._disc_optimizer, device)

    def _conditional_loss(
        self,
        data: torch.Tensor,
        conditional_vectors: torch.Tensor,
        mask_vectors: torch.Tensor,
        metadata: MetaData,
    ) -> torch.Tensor:
        """Compute the cross-entropy loss on the fixed categorical columns."""
        loss = []
        start = 0
        start_cond = 0
        for colum_meta in metadata:
            if colum_meta.column_type != DataTypes.CATEGORICAL:
                start += colum_meta.output_dimension
            else:
                end = start + colum_meta.output_dimension
                end_cond = start_cond + colum_meta.output_dimension

                # calculate cross entropy loss between the spec. columns and
                # a subset of the conditional vector
                tmp = functional.cross_entropy(
                    data[:, start:end],
                    # which category in the categorical column is 1 (or 0 if not
                    # conditioned on this categorical column)
                    torch.argmax(conditional_vectors[:, start_cond:end_cond], dim=1),
                    reduction="none",
                )
                # collect the loss for each categorical column
                loss.append(tmp)
                start = end
                start_cond = end_cond

        loss = torch.stack(loss, dim=1)

        # return the loss per batch size, not per data point + apply mask_vectors
        return (loss * mask_vectors).sum() / data.size()[0]

    def _init_model_and_optim(self):
        """Initialize the model and optimizers."""
        data_dim = self.metadata.num_transformed_columns()

        # to include the metadata in the activation_fn
        generator_activation_fn = partial(apply_activations, metadata=self.metadata)

        self.model = CTGAN(
            data_dim=data_dim,
            cond_dim=self.sampler.dim_conditional_vector(),
            embedding_dim=self.config["embedding_dim"],
            generator_dim=self.config["generator_dim"],
            discriminator_dim=self.config["discriminator_dim"],
            pac=self.config["pac"],
            generator_activation_fn=generator_activation_fn,
        )

        self._gen_optimizer = optim.Adam(
            self.model.generator.parameters(),
            lr=self.config["generator_lr"],
            betas=self.config["betas"],
            weight_decay=self.config["generator_decay"],
        )

        self._disc_optimizer = optim.Adam(
            self.model.discriminator.parameters(),
            lr=self.config["discriminator_lr"],
            betas=self.config["betas"],
            weight_decay=self.config["discriminator_decay"],
        )


class SimpleCtganSynthesizer(CtganSynthesizer):
    """Conditional Table GAN Synthesizer.

    Orchestrates different components to model tabular data using Conditional GAN.
    """

    metadata: Optional[MetaData] = None
    transformer: SimpleTransformer
    sampler: Optional[DataSampler] = None
    model: Optional[CTGAN] = None
    epoch: int = 0

    def __init__(
        self,
        embedding_dim: int = 128,
        generator_dim: Sequence[int] = (256, 256),
        discriminator_dim: Sequence[int] = (256, 256),
        generator_lr: float = 2e-4,
        generator_decay: float = 1e-6,
        discriminator_lr: float = 2e-4,
        discriminator_decay: float = 1e-6,
        betas: Tuple[float, float] = (0.5, 0.9),
        batch_size: int = 500,
        discriminator_steps: int = 1,
        gradient_penalty_influence: float = 10.0,
        log_frequency: bool = True,
        pac: int = 10,
    ):
        """Initialize the CTGAN Synthesizer with the given parameters."""
        assert batch_size % pac == 0

        self.config = {
            "embedding_dim": embedding_dim,
            "generator_dim": generator_dim,
            "discriminator_dim": discriminator_dim,
            "generator_lr": generator_lr,
            "generator_decay": generator_decay,
            "discriminator_lr": discriminator_lr,
            "discriminator_decay": discriminator_decay,
            "betas": betas,
            "batch_size": batch_size,
            "discriminator_steps": discriminator_steps,
            "gradient_penalty_influence": gradient_penalty_influence,
            "log_frequency": log_frequency,
            "pac": pac,
            "epoch": 0,
        }

        self.transformer = SimpleTransformer()

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load the state dictionary into the synthesizer."""
        self.config = state_dict["config"]
        self.epoch = self.config["epoch"]
        self.transformer = SimpleTransformer()
        self.transformer.load_state_dict(state_dict["transformer_state_dict"])
        self.metadata = self.transformer.metadata
        self.sampler = DataSampler()
        self.sampler.load_state_dict(state_dict["sampler_state_dict"])

        self._init_model_and_optim()
        self.model.load_state_dict(state_dict["model_state_dict"])
        self._gen_optimizer.load_state_dict(state_dict["gen_optimizer_state_dict"])
        self._disc_optimizer.load_state_dict(state_dict["disc_optimizer_state_dict"])
