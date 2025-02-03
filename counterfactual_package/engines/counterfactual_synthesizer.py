"""
This module contains classes and functions to handle Counterfactual Generative
Adversarial Networks (CFGAN) synthesizers. These classes are designed for training
and generating counterfactuals using CFGANS with different configurations and
ablation studies.
"""

import logging
from functools import partial
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.data_preprocess import infer_continuous_columns
from ..data.datasets import OnlyFeaturesDataset, PandasDataset, prepare_dataloaders
from ..data.metadata import MetaData
from ..data.sampler import DataSampler
from ..data.transformer import (
    CtganTransformer,
    adjust_metadata_for_simple_transform,
    reverse_simple_transform,
    simple_transform,
)
from ..data.utils import (
    apply_mask,
    calculate_distances,
    generate_counterfactual_masks,
    randomly_change_to_other_cat,
    randomly_get_other_value_ids,
    transform_mask,
)
from ..models.gans import FlexibleCounterFactualGan
from ..utils.bookkeeping import Accumulator
from ..utils.utils import move_optimizer_to_device
from .activations import apply_activations
from .ctgan_synthesizer import CtganSynthesizer
from .losses import calc_gradient_penalty, masked_divergence_loss

logger = logging.getLogger(__name__)


class FCEGAN:
    """
    FCEGAN class for generating counterfactual examples using counterfactual templates.

    Attributes:
        metadata (Optional[MetaData]): Metadata for the dataset.
        transformer (CtganTransformer): Transformer for data preprocessing.
        sampler (DataSampler): Sampler for generating synthetic data.
        transform (Callable[[Union[pd.DataFrame, pd.Series]], torch.Tensor]): Function
            to transform data.
        reverse_transform (Callable[[torch.Tensor], Union[pd.DataFrame, np.ndarray]]):
            Function to reverse transform data.
        model (FlexibleCounterFactualGan): GAN model for counterfactual generation.
        epoch (int): Current training epoch.
        _disc_optimizer (optim.Optimizer): Optimizer for the discriminator.
        _gen_optimizer (optim.Optimizer): Optimizer for the generator.

    Methods:
        __init__
        initialize_training
        fit_epoch
        generate_valid_counterfactuals
        evaluate_counterfactuals
        load_state_dict
        state_dict
        to
        post_process_counterfactuals
        _post_process_transformed_counterfactuals
        _train_step
        _test_step
        _disc_step
        _disc_update
        _gen_step
        _gen_update
        _init_model_and_optim
    """

    metadata: Optional[MetaData] = None
    transformer: CtganTransformer = None
    sampler: DataSampler
    transform: Callable[[Union[pd.DataFrame, pd.Series]], torch.Tensor]
    reverse_transform: Callable[[torch.Tensor], Union[pd.DataFrame, np.ndarray]]
    classifier: Optional[nn.Module]

    model: FlexibleCounterFactualGan
    epoch: int = 0

    _disc_optimizer: optim.Optimizer
    _gen_optimizer: optim.Optimizer

    def __init__(
        self,
        embedding_dim: int = 128,
        discriminator_dim: Sequence[int] = (256, 256),
        generator_dim: Sequence[int] = (256, 256),
        discriminator_steps: int = 1,
        pac: int = 10,
        cf_dropout_range: Tuple[float, float] = (0.5, 1.0),
        discriminator_lr: float = 2e-4,
        generator_lr: float = 2e-4,
        discriminator_decay: float = 1e-6,
        generator_decay: float = 1e-6,
        betas: Tuple[float, float] = (0.5, 0.9),
        gradient_penalty_influence: float = 10.0,
        continuous_float_influence: float = 1.0,
        counterfactual_disc_influence: float = 0.5,
        original_disc_influence: float = 0.5,
        reconstruction_divergence_influence: float = 1.0,
        cf_divergence_influence: float = 0.0,
        classifier_influence: float = 1.0,
        **kwargs,
    ) -> None:
        """
        Initialize the CFGAN synthesizer with given parameters.

        Args:
            embedding_dim (int): Dimension of the embedding space.
            discriminator_dim (Sequence[int]): Dimensions of the discriminator layers.
            generator_dim (Sequence[int]): Dimensions of the generator layers.
            discriminator_steps (int): Steps of discriminator training per generator \
                step.
            pac (int): Number of samples fed into the discriminator.
            cf_dropout_range (Tuple[float, float]): Range for dropout rates in \
                counterfactual generation.
            discriminator_lr (float): Learning rate for discriminator.
            generator_lr (float): Learning rate for generator.
            discriminator_decay (float): Weight decay for discriminator.
            generator_decay (float): Weight decay for generator.
            betas (Tuple[float, float]): Betas for Adam optimizer.
            gradient_penalty_influence (float): Influence of the gradient penalty.
            continuous_float_influence (float): Influence for continuous float values.
            counterfactual_disc_influence (float): Influence of counterfactual \
                discriminator loss.
            original_disc_influence (float): Influence of original discriminator loss.
            reconstruction_divergence_influence (float): Influence of reconstruction \
                divergence loss.
            cf_divergence_influence (float): Influence of counterfactual divergence \
                loss.
            classifier_influence (float): Influence of classifier loss.
        """

        self.config = {
            "embedding_dim": embedding_dim,
            "discriminator_dim": discriminator_dim,
            "generator_dim": generator_dim,
            "discriminator_steps": discriminator_steps,
            "pac": pac,
            "cf_dropout_range": cf_dropout_range,
            "discriminator_lr": discriminator_lr,
            "generator_lr": generator_lr,
            "discriminator_decay": discriminator_decay,
            "generator_decay": generator_decay,
            "betas": betas,
            "gradient_penalty_influence": gradient_penalty_influence,
            "continuous_float_influence": continuous_float_influence,
            "counterfactual_disc_influence": counterfactual_disc_influence,
            "original_disc_influence": original_disc_influence,
            "reconstruction_divergence_influence": reconstruction_divergence_influence,
            "cf_divergence_influence": cf_divergence_influence,
            "classifier_influence": classifier_influence,
            "epoch": 0,
        }
        self.transformer = CtganTransformer()
        self.sampler = DataSampler()

    def initialize_training(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        target_name: str,
        classifier: Optional[nn.Module] = None,
        batch_size: int = 300,
        state_dict: Optional[Dict[str, Any]] = None,
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Initialize training with the given training and test data.

        Args:
            train (pd.DataFrame): Training data.
            test (pd.DataFrame): Test data.
            target_name (str): Target column name.
            classifier (Optional[nn.Module]): Pre-trained classifier.
            batch_size (int): Batch size.
            state_dict (Optional[Dict[str, Any]]): State dictionary for loading \
                model state.

        Returns:
            Tuple[DataLoader, DataLoader]: Data loaders for training and test data.
        """

        logger.info("Initializing for training.")

        self.config["target_name"] = target_name

        if classifier:
            self.classifier = classifier

        if state_dict:
            self.load_state_dict(state_dict)

        if batch_size > len(test) or batch_size > len(train):
            raise ValueError(f"Batch size of {batch_size} too big.")
        elif batch_size % self.config["pac"]:
            raise ValueError("Batch size has to be divisible by pac size")

        train_loader, test_loader, self.transformer = prepare_dataloaders(
            train=train,
            test=test,
            data_transformer=self.transformer,
            batch_size=batch_size,
        )

        self.transform = self.transformer.transform
        self.reverse_transform = self.transformer.reverse_transform

        if not state_dict:
            self.metadata = self.transformer.metadata
            train_tf = self.transform(train)
            self.sampler.fit(train_tf, self.metadata)
            self._init_model_and_optim()

        return train_loader, test_loader

    def fit_epoch(
        self, train_loader: DataLoader, test_loader: DataLoader, device: torch.device
    ) -> Dict[str, Any]:
        """
        Fit the model for one epoch.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            test_loader (DataLoader): DataLoader for test data.
            device (torch.device): Device to use for training.

        Returns:
            Dict[str, Any]: Training and test losses.
        """
        train_losses = self._train_step(train_loader, device)
        test_losses = self._test_step(test_loader, device)
        self.epoch += 1

        return {"epoch": self.epoch, **train_losses, **test_losses}

    def generate_valid_counterfactuals(
        self,
        original: Union[pd.DataFrame, pd.Series],
        counterfactual_template: Union[pd.DataFrame, pd.Series],
        n: int,
        classifier: nn.Module,
        ctgan: CtganSynthesizer,
        max_iterations: int = 1000,
        batch_size: int = 300,
        sort: bool = True,
    ) -> Optional[pd.DataFrame]:
        """
        Generate valid counterfactuals given an original and template.

        Args:
            original (Union[pd.DataFrame, pd.Series]): Original sample.
            counterfactual_template (Union[pd.DataFrame, pd.Series]): Template for \
                counterfactual.
            n (int): Number of counterfactuals to generate.
            classifier (nn.Module): Classifier to validate counterfactuals.
            ctgan (CtganSynthesizer): Pre-trained CTGAN synthesizer.
            max_iterations (int): Maximum number of iterations.
            batch_size (int): Batch size.
            sort (bool): Whether to sort counterfactuals by certainty.

        Returns:
            Optional[pd.DataFrame]: Valid counterfactuals.
        """
        if isinstance(original, pd.Series):
            original = original.to_frame().T

        if isinstance(counterfactual_template, pd.Series):
            counterfactual_template = counterfactual_template.to_frame().T
        # repeat 'batch_size' times to batch process
        repeated_originals = pd.concat([original] * batch_size, ignore_index=True)
        originals_tensor = self.transform(repeated_originals)

        repeated_cf_templates = pd.concat(
            [counterfactual_template] * batch_size, ignore_index=True
        )
        to_change_mask = transform_mask(
            torch.tensor(repeated_cf_templates.isna().to_numpy(), dtype=torch.bool),
            self.metadata,
        )
        cf_templates_tensor = originals_tensor.masked_fill(to_change_mask, 0)

        # change to right counterfactual target
        target_idxs = self.metadata.column_to_transformed_idxs(
            self.config["target_name"]
        )
        cf_targets = self.transform(repeated_cf_templates[self.config["target_name"]])
        cf_templates_tensor[:, target_idxs] = cf_targets

        # collect valid counterfactuals
        valid_counterfactuals = pd.DataFrame(columns=(list(original.columns))).dropna(
            axis=1, how="all"
        )
        iteration = 0
        with tqdm(total=n) as pbar:
            while len(valid_counterfactuals) < n and iteration <= max_iterations:
                iteration += 1

                raw_counterfactual_tensor = self.model.predict(
                    originals_tensor, cf_templates_tensor
                )
                raw_counterfactuals = self.reverse_transform(raw_counterfactual_tensor)
                counterfactuals = self.post_process_counterfactuals(
                    raw_counterfactuals, repeated_cf_templates
                )

                metrics = self.evaluate_counterfactuals(
                    counterfactuals, repeated_originals, classifier, ctgan, batch_size
                )["global_metrics"]

                # only gather the valid samples
                valid = metrics["valid_counterfactuals"]

                if valid.sum() != 0:
                    new_valid_counterfactuals = pd.concat(
                        [
                            counterfactuals[valid],
                            metrics[valid],
                        ],
                        axis=1,
                    )
                    valid_counterfactuals = pd.concat(
                        [
                            valid_counterfactuals,
                            new_valid_counterfactuals,
                        ],
                        axis=0,
                    )

                pbar.n = len(valid_counterfactuals)
                pbar.refresh()

        if len(valid_counterfactuals) == 0:
            print("No valid samples were generated. Try again!")
            return valid_counterfactuals
        if sort:
            valid_counterfactuals = valid_counterfactuals.sort_values(
                ["counterfactual_pred"], ascending=False
            )
        if len(valid_counterfactuals) > n:
            valid_counterfactuals = valid_counterfactuals[:n]

        print(
            f"Out of {iteration * batch_size} proposals "
            f"{len(valid_counterfactuals)}/{n} samples of good quality."
        )
        return valid_counterfactuals

    def evaluate_counterfactuals(
        self,
        counterfactuals: pd.DataFrame,
        originals: pd.DataFrame,
        classifier: nn.Module,
        ctgan: CtganSynthesizer,
        batch_size: int = 300,
    ):
        """
        Evaluate the generated counterfactuals.

        Args:
            counterfactuals (pd.DataFrame): Generated counterfactuals.
            originals (pd.DataFrame): Original samples.
            classifier (nn.Module): Classifier to validate counterfactuals.
            ctgan (CtganSynthesizer): Pre-trained CTGAN synthesizer.
            batch_size (int): Batch size.

        Returns:
            Dict[str, Any]: Evaluation metrics.
        """
        # 1) classifier target shift
        cf_dataset = PandasDataset(
            counterfactuals,
            self.config["target_name"],
            self.transform,
        )
        originals_set = PandasDataset(
            originals,
            self.config["target_name"],
            self.transform,
        )

        cf_loader = DataLoader(cf_dataset, batch_size=batch_size)
        original_loader = DataLoader(originals_set, batch_size=batch_size)

        prediction_gains = []
        cf_predictions = []
        valid_counterfactuals = []

        classifier.eval()
        with torch.no_grad():
            for (cf_features, cf_targets), (og_features, _) in zip(
                cf_loader, original_loader
            ):
                cf_logits = classifier(cf_features)
                og_logits = classifier(og_features)

                cf_prediction = F.softmax(cf_logits, dim=1)[
                    range(cf_features.shape[0]), cf_targets
                ]
                original_prediction = F.softmax(og_logits, dim=1)[
                    range(cf_features.shape[0]), cf_targets
                ]

                prediction_gains.append(cf_prediction - original_prediction)
                cf_predictions.append(cf_prediction)

                # Check if the argmax is effectively the cf_targets
                cf_argmax = torch.argmax(cf_logits, dim=1)
                correct_predictions = (cf_argmax == cf_targets).float()
                valid_counterfactuals.append(correct_predictions)

            prediction_gains = torch.cat(prediction_gains, dim=0)
            cf_predictions = torch.cat(cf_predictions, dim=0)
            valid_counterfactuals = torch.cat(valid_counterfactuals, dim=0)

        # 2) divergence metrics
        unmasked_counterfactual_templates = originals.copy()
        unmasked_counterfactual_templates[self.config["target_name"]] = counterfactuals[
            self.config["target_name"]
        ]
        distances = calculate_distances(
            unmasked_counterfactual_templates, counterfactuals, self.metadata
        )
        cat_changed = np.mean(distances["hamming"], axis=1)
        mean_percentile_shift = distances["percentile_shifts"].abs().mean(axis=1)
        max_percentile_shift = distances["percentile_shifts"].abs().max(axis=1)

        # 3) include realism - discriminator
        target_name = self.config["target_name"]
        # Make conditional vectors
        col_id = ctgan.metadata.column_to_idx(target_name, only_categories=True)
        cond_vector_dim = ctgan.sampler.dim_conditional_vector()
        cond_vector_start = ctgan.sampler.categorical_column_cond_start_id[col_id]
        value_ids = torch.argmax(
            ctgan.transformer.transform(counterfactuals[target_name]), dim=1
        )
        cond_vecs = torch.zeros((len(value_ids), cond_vector_dim))
        cond_vecs[torch.arange(len(value_ids)), cond_vector_start + value_ids] = 1.0

        # TODO: make batches
        counterfactuals = ctgan.transformer.transform(counterfactuals)
        discriminator_input = torch.cat([counterfactuals, cond_vecs], dim=1)
        # Discriminator works per pacs
        discriminator_input_repeated = discriminator_input.repeat_interleave(
            ctgan.config["pac"], dim=0
        )
        with torch.no_grad():
            fakeness = ctgan.model.discriminator(discriminator_input_repeated).squeeze()

        global_metrics = pd.DataFrame(
            {
                "cat_changed": cat_changed,
                "mean_percentile_shift": mean_percentile_shift,
                "max_percentile_shift": max_percentile_shift,
                "counterfactual_pred": cf_predictions,
                "prediction_gain": prediction_gains,
                "valid_counterfactuals": valid_counterfactuals,
                "fakeness": fakeness,
            }
        )

        return {**distances, "global_metrics": global_metrics}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """
        Load the state dictionary.

        Args:
            state_dict (Dict[str, Any]): State dictionary to load.
        """
        self.config = state_dict["config"]
        self.epoch = self.config["epoch"]

        self.transformer = CtganTransformer()
        self.transformer.load_state_dict(state_dict["transformer_state_dict"])
        self.metadata = self.transformer.metadata

        self.transform = self.transformer.transform
        self.reverse_transform = self.transformer.reverse_transform

        self.sampler = DataSampler()
        self.sampler.load_state_dict(state_dict["sampler_state_dict"])

        self._init_model_and_optim()
        self.model.load_state_dict(state_dict["model_state_dict"])
        self._gen_optimizer.load_state_dict(state_dict["gen_optimizer_state_dict"])
        self._disc_optimizer.load_state_dict(state_dict["disc_optimizer_state_dict"])

    def state_dict(self) -> Dict[str, Any]:
        """
        Get the state dictionary of the model.

        Returns:
            Dict[str, Any]: State dictionary of the model.
        """
        self.config["epoch"] = self.epoch
        return {
            "model_state_dict": self.model.state_dict(),
            "gen_optimizer_state_dict": self._gen_optimizer.state_dict(),
            "disc_optimizer_state_dict": self._disc_optimizer.state_dict(),
            "transformer_state_dict": self.transformer.state_dict(),
            "sampler_state_dict": self.sampler.state_dict(),
            "config": self.config,
        }

    def to(self, device: torch.device):
        """
        Move the model and optimizers to the specified device.

        Args:
            device (torch.device): Device to move the model and optimizers to.
        """
        self.model.to(device)
        move_optimizer_to_device(self._gen_optimizer, device)
        move_optimizer_to_device(self._disc_optimizer, device)

    def post_process_counterfactuals(
        self,
        raw_counterfactuals: pd.DataFrame,
        counterfactual_templates: pd.DataFrame,
    ):
        """
        Post-process generated counterfactuals.

        Args:
            raw_counterfactuals (pd.DataFrame): Raw generated counterfactuals.
            counterfactual_templates (pd.DataFrame): Counterfactual templates.

        Returns:
            pd.DataFrame: Post-processed counterfactuals.
        """
        return counterfactual_templates.reset_index(drop=True).fillna(
            raw_counterfactuals.reset_index(drop=True)
        )

    def _post_process_transformed_counterfactuals(
        self,
        raw_counterfactuals: torch.Tensor,
        counterfactual_templates: torch.Tensor,
    ):
        """
        Post-process transformed counterfactuals by applying a mask to the raw\
              counterfactuals.

        This method takes raw generated counterfactuals and counterfactual templates,
        and processes them to produce the final counterfactuals. The process involves
        creating a mask that identifies which parts of the raw counterfactuals
        should be changed based on the counterfactual templates. The mask is then 
        applied to the raw counterfactuals to generate the final post-processed
        counterfactuals.
        
        Args:
            raw_counterfactuals (torch.Tensor): A tensor containing the raw generated\
                  counterfactuals.
            counterfactual_templates (torch.Tensor): A tensor containing the\
                  counterfactual templates.
        Returns:
            torch.Tensor: A tensor containing the post-processed counterfactuals, where the raw counterfactuals
                  have been selectively applied to the templates based on the mask.
        """
        # make mask
        to_change_mask = []
        start = 0
        for col_meta in self.metadata:
            output_dim = col_meta.output_dimension
            end = start + output_dim
            to_change_col = counterfactual_templates[:, start:end].sum(dim=1) == 0
            to_change_mask.append(to_change_col.unsqueeze(1).repeat(1, output_dim))
            start = end

        to_change_mask = torch.cat(to_change_mask, dim=1)

        # make counterfactuals
        counterfactuals = counterfactual_templates + apply_mask(
            tensor=raw_counterfactuals, mask=to_change_mask
        )
        return counterfactuals

    def _train_step(
        self, train_dataloader: DataLoader, device: torch.device
    ) -> Dict[str, Any]:
        """
        Perform a training step.

        Args:
            train_dataloader (DataLoader): DataLoader for training data.
            device (torch.device): Device to use for training.

        Returns:
            Dict[str, Any]: Training losses.
        """

        self.model.train()

        loss_accumulator = Accumulator(6)

        for originals in train_dataloader:
            originals = originals.to(device)

            for _ in range(self.config["discriminator_steps"]):
                disc_loss = self._disc_step(originals, device)
                self._disc_update(disc_loss)

            disc_output_loss, reconst_div_loss, cf_div_loss, classifier_loss = (
                self._gen_step(originals)
            )
            gen_loss = (
                disc_output_loss + reconst_div_loss + cf_div_loss + classifier_loss
            )
            self._gen_update(gen_loss)

            loss_accumulator.add(
                disc_loss.detach().cpu(),
                gen_loss.detach().cpu(),
                disc_output_loss.detach().cpu(),
                reconst_div_loss.detach().cpu(),
                cf_div_loss.detach().cpu(),
                classifier_loss.detach().cpu(),
            )

        losses = loss_accumulator / len(train_dataloader)

        return {
            "train_disc_loss": losses[0],
            "train_gen_loss": losses[1],
            "train_gen_by_disc_loss": losses[2],
            "train_reconst_divergence_loss": losses[3],
            "train_cf_divergence_loss": losses[4],
            "train_classifier_loss": losses[5],
        }

    def _test_step(
        self, test_dataloader: DataLoader, device: torch.device
    ) -> Dict[str, Any]:
        """
        Perform a test step.

        Args:
            test_dataloader (DataLoader): DataLoader for test data.
            device (torch.device): Device to use for testing.

        Returns:
            Dict[str, Any]: Test losses.
        """

        self.model.eval()

        loss_accumulator = Accumulator(6)

        with torch.no_grad():
            for originals in test_dataloader:
                originals = originals.to(device)

                disc_loss = self._disc_step(originals, device)

                disc_output_loss, reconst_div_loss, cf_div_loss, classifier_loss = (
                    self._gen_step(originals)
                )
                gen_loss = (
                    disc_output_loss + reconst_div_loss + cf_div_loss + classifier_loss
                )

                loss_accumulator.add(
                    disc_loss.detach().cpu(),
                    gen_loss.detach().cpu(),
                    disc_output_loss.detach().cpu(),
                    reconst_div_loss.detach().cpu(),
                    cf_div_loss.detach().cpu(),
                    classifier_loss.detach().cpu(),
                )

        losses = loss_accumulator / len(test_dataloader)

        return {
            "test_disc_loss": losses[0],
            "test_gen_loss": losses[1],
            "test_gen_by_disc_loss": losses[2],
            "test_reconst_divergence_loss": losses[3],
            "test_cf_divergence_loss": losses[4],
            "test_classifier_loss": losses[5],
        }

    def _disc_step(self, originals: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Perform a discriminator step.

        Args:
            originals (torch.Tensor): Original samples.
            device (torch.device): Device to use for training.

        Returns:
            torch.Tensor: Discriminator loss.
        """
        batch_size = originals.shape[0]

        keep_cols = generate_counterfactual_masks(
            dropout=self.config["cf_dropout_range"],
            batch_size=batch_size,
            no_dropout_cols=[self.config["target_name"]],
            metadata=self.metadata,
        )
        keep_transformed = transform_mask(mask=keep_cols, metadata=self.metadata)
        masked_originals = apply_mask(tensor=originals, mask=keep_transformed)

        # randomly select other target value as counterfactual
        counterfactual_templates = randomly_change_to_other_cat(
            data=masked_originals,
            column=self.config["target_name"],
            metadata=self.metadata,
        )

        noise = torch.normal(
            mean=0, std=1, size=(batch_size, self.config["embedding_dim"])
        )
        generator_input = torch.cat([originals, counterfactual_templates, noise], dim=1)
        raw_counterfactuals = self.model.generator(generator_input)
        counterfactuals = self._post_process_transformed_counterfactuals(
            raw_counterfactuals, counterfactual_templates
        )

        # Sample real counterfactuals
        target_cat_col_id = self.metadata.column_to_idx(
            self.config["target_name"], only_categories=True
        )
        target_cat_col_ids = torch.full((batch_size,), target_cat_col_id)
        transformed_target_idxs = self.metadata.column_to_transformed_idxs(
            self.config["target_name"]
        )
        value_ids = counterfactuals[:, transformed_target_idxs].argmax(dim=1)
        real_counterfactuals = self.sampler.sample_data(
            cat_col_ids=target_cat_col_ids, value_ids=value_ids
        )

        # only original/counterfactual fed in -> real/fake
        disc_originals_pred = self.model.discriminator(originals).reshape(-1)
        disc_real_counterfactual_pred = self.model.discriminator(real_counterfactuals)
        disc_fake_pred = self.model.discriminator(counterfactuals).reshape(-1)

        gp = calc_gradient_penalty(
            real_data=originals,
            fake_data=counterfactuals,
            discriminator=self.model.discriminator,
            pac=self.config["pac"],
            device=device,
            gradient_penalty_influence=self.config["gradient_penalty_influence"],
        )

        original_disc_loss = -(
            torch.mean(disc_originals_pred) - torch.mean(disc_fake_pred)
        )
        counterfactual_disc_loss = -(
            torch.mean(disc_real_counterfactual_pred) - torch.mean(disc_fake_pred)
        )

        return (
            original_disc_loss * self.config["original_disc_influence"]
            + counterfactual_disc_loss * self.config["counterfactual_disc_influence"]
            + gp
        )

    def _disc_update(self, disc_loss: torch.Tensor) -> None:
        """
        Update the discriminator.

        Args:
            disc_loss (torch.Tensor): Discriminator loss.
        """
        self._disc_optimizer.zero_grad()
        disc_loss.backward()
        self._disc_optimizer.step()

    def _gen_step(
        self, originals: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform a generator step.

        Args:
            originals (torch.Tensor): Original samples.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Generator \
                losses.
        """
        batch_size = originals.shape[0]

        keep_cols = generate_counterfactual_masks(
            dropout=self.config["cf_dropout_range"],
            batch_size=batch_size,
            no_dropout_cols=[self.config["target_name"]],
            metadata=self.metadata,
        )
        to_change = ~keep_cols
        keep_transformed = transform_mask(mask=keep_cols, metadata=self.metadata)

        # randomly select other target value as counterfactual
        unmasked_counterfactual_templates = randomly_change_to_other_cat(
            data=originals,
            column=self.config["target_name"],
            metadata=self.metadata,
        )
        counterfactual_templates = apply_mask(
            tensor=unmasked_counterfactual_templates, mask=keep_transformed
        )

        noise = torch.normal(
            mean=0, std=1, size=(batch_size, self.config["embedding_dim"])
        )
        generator_input = torch.cat([originals, counterfactual_templates, noise], dim=1)
        raw_counterfactuals = self.model.generator(generator_input)
        counterfactuals = self._post_process_transformed_counterfactuals(
            raw_counterfactuals, counterfactual_templates
        )

        disc_output_loss = torch.mean(-self.model.discriminator(counterfactuals))

        # normalize by number of features (and batch size)
        reconstruction_loss = masked_divergence_loss(
            predicted=raw_counterfactuals,
            target=counterfactual_templates,
            col_mask=keep_cols,
            metadata=self.metadata,
            continous_float_influence=self.config["continuous_float_influence"],
        ).mean()

        # normalize by number of features (and batch size)
        counterfactual_loss = masked_divergence_loss(
            predicted=counterfactuals,
            target=originals,
            col_mask=to_change,
            metadata=self.metadata,
            continous_float_influence=self.config["continuous_float_influence"],
        ).mean()

        disc_influence = (
            self.config["original_disc_influence"]
            + self.config["counterfactual_disc_influence"]
        )

        # classifier loss
        if self.classifier:
            target_col_idxs = self.metadata.column_to_transformed_idxs(
                self.config["target_name"]
            )
            feature_col_idxs = list(
                set(range(counterfactuals.shape[1])) - set(target_col_idxs)
            )
            target = torch.argmax(counterfactuals[:, target_col_idxs], dim=1)
            counterfactual_features = counterfactuals[:, feature_col_idxs]
            target_pred = self.classifier(counterfactual_features)
            classifier_loss = F.cross_entropy(target_pred, target).mean()
        else:
            classifier_loss = torch.tensor(0)

        return (
            disc_output_loss * disc_influence,
            reconstruction_loss * self.config["reconstruction_divergence_influence"],
            counterfactual_loss * self.config["cf_divergence_influence"],
            classifier_loss * self.config["classifier_influence"],
        )

    def _gen_update(self, gen_loss: torch.Tensor) -> None:
        """
        Update the generator.

        Args:
            gen_loss (torch.Tensor): Generator loss.
        """
        self._gen_optimizer.zero_grad()
        gen_loss.backward()
        self._gen_optimizer.step()

    def _init_model_and_optim(self):
        """
        Initialize the model and optimizers.
        """
        if not self.metadata:
            raise ValueError("Metadata unknown. First prepare the transformer.")

        data_dim = self.metadata.num_transformed_columns()

        # to include the metadata in the activation_fn
        generator_activation_fn = partial(apply_activations, metadata=self.metadata)

        self.model = FlexibleCounterFactualGan(
            data_dim=data_dim,
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


class SimpleFCEGAN(FCEGAN):
    """
    SimpleFCEGAN is a class that extends the FCEGAN class to provide a simple
    data transformation instead of the CTGAN data transformation. This results
    in a simpler design to illustrate FCEGAN is not limited to CTGAN transformations.

    Rewritten Methods:
        initialize_training
        load_state_dict
    """

    def initialize_training(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        target_name: str,
        classifier: Optional[nn.Module] = None,
        batch_size: int = 300,
        state_dict: Optional[Dict[str, Any]] = None,
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Initialize training with the given training and test data.

        Args:
            train (pd.DataFrame): Training data.
            test (pd.DataFrame): Test data.
            target_name (str): Target column name.
            classifier (Optional[nn.Module]): Pre-trained classifier.
            batch_size (int): Batch size.
            state_dict (Optional[Dict[str, Any]]): State dictionary for loading \
                model state.

        Returns:
            Tuple[DataLoader, DataLoader]: Data loaders for training and test data.
        """
        logger.info("Initializing for training.")

        self.config["target_name"] = target_name

        if classifier:
            self.classifier = classifier

        if state_dict:
            self.load_state_dict(state_dict)

        if batch_size > len(test) or batch_size > len(train):
            raise ValueError(f"Batch size of {batch_size} too big.")
        elif batch_size % self.config["pac"]:
            raise ValueError("Batch size has to be divisible by pac size")

        features = train.columns.to_list()
        continuous_cols = infer_continuous_columns(train)
        categorical_cols = list(set(features) - set(continuous_cols))

        if not self.transformer.metadata:
            self.transformer.fit(raw_data=train, categorical_columns=categorical_cols)
        self.metadata = adjust_metadata_for_simple_transform(self.transformer.metadata)

        self.transform = partial(simple_transform, metadata=self.metadata)
        self.reverse_transform = partial(
            reverse_simple_transform, metadata=self.metadata
        )

        train_transformed = self.transform(train)
        test_transformed = self.transform(test)

        train_dataset = OnlyFeaturesDataset(train_transformed)
        test_dataset = OnlyFeaturesDataset(test_transformed)

        train_dataloader = DataLoader(
            train_dataset, batch_size, shuffle=True, drop_last=True
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size, shuffle=True, drop_last=True
        )

        if not state_dict:
            train_tf = self.transform(train)
            self.sampler.fit(train_tf, self.metadata)
            self._init_model_and_optim()

        return train_dataloader, test_dataloader

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """
        Load the state dictionary.

        Args:
            state_dict (Dict[str, Any]): State dictionary to load.
        """
        self.config = state_dict["config"]
        self.epoch = self.config["epoch"]

        self.transformer = CtganTransformer()
        self.transformer.load_state_dict(state_dict["transformer_state_dict"])
        self.metadata = adjust_metadata_for_simple_transform(self.transformer.metadata)

        self.transform = partial(simple_transform, metadata=self.metadata)
        self.reverse_transform = partial(
            reverse_simple_transform, metadata=self.metadata
        )

        self.sampler = DataSampler()
        self.sampler.load_state_dict(state_dict["sampler_state_dict"])

        self._init_model_and_optim()
        self.model.load_state_dict(state_dict["model_state_dict"])
        self._gen_optimizer.load_state_dict(state_dict["gen_optimizer_state_dict"])
        self._disc_optimizer.load_state_dict(state_dict["disc_optimizer_state_dict"])


class NoTemplateFCEGAN(FCEGAN):
    """
    NoTemplateFCEGAN is a class that extends the FCEGAN class to generate valid
    counterfactuals without using a counterfactual template.

    This purely serves as an ablation model. The counterfactual template is
    replaced by a copy of the sample. This is necessary for a fair comparison
    since the neural network will have a comparable model size.
    A counterfactual template is still being generated for the post-processing
    step but not fed into the generator.

    One could additionally set the 'cf_dropout_range' to (1.0, 1.0) to train a
    vanilla counterfactual generative model. This was done in the paper for the
    'No Template' model.

    Rewritten Methods:
        generate_valid_counterfactuals
        _disc_step
        _disc_update
        _gen_step
    """

    def generate_valid_counterfactuals(
        self,
        original: Union[pd.DataFrame, pd.Series],
        counterfactual_template: Union[pd.DataFrame, pd.Series],
        n: int,
        classifier: nn.Module,
        ctgan: CtganSynthesizer,
        max_iterations: int = 1000,
        batch_size: int = 300,
        sort: bool = True,
    ) -> Optional[pd.DataFrame]:
        """
        Generate valid counterfactuals given an original and template.

        Args:
            original (Union[pd.DataFrame, pd.Series]): Original sample.
            counterfactual_template (Union[pd.DataFrame, pd.Series]): Template for \
                counterfactual.
            n (int): Number of counterfactuals to generate.
            classifier (nn.Module): Classifier to validate counterfactuals.
            ctgan (CtganSynthesizer): Pre-trained CTGAN synthesizer.
            max_iterations (int): Maximum number of iterations.
            batch_size (int): Batch size.
            sort (bool): Whether to sort counterfactuals by certainty.

        Returns:
            Optional[pd.DataFrame]: Valid counterfactuals.
        """
        if isinstance(original, pd.Series):
            original = original.to_frame().T

        if isinstance(counterfactual_template, pd.Series):
            counterfactual_template = counterfactual_template.to_frame().T

        # repeat 'batch_size' times to batch process
        repeated_originals = pd.concat([original] * batch_size, ignore_index=True)
        originals_tensor = self.transform(repeated_originals)

        repeated_cf_templates = pd.concat(
            [counterfactual_template] * batch_size, ignore_index=True
        )
        to_change_mask = transform_mask(
            torch.tensor(repeated_cf_templates.isna().to_numpy(), dtype=torch.bool),
            self.metadata,
        )
        cf_templates_tensor = originals_tensor.masked_fill(to_change_mask, 0)

        # change to right counterfactual target
        target_idxs = self.transformer.metadata.column_to_transformed_idxs(
            self.config["target_name"]
        )
        cf_targets = self.transform(repeated_cf_templates[self.config["target_name"]])
        cf_templates_tensor[:, target_idxs] = cf_targets

        # collect valid counterfactuals
        valid_counterfactuals = pd.DataFrame(columns=(list(original.columns))).dropna(
            axis=1, how="all"
        )
        iteration = 0
        with tqdm(total=n) as pbar:
            while len(valid_counterfactuals) < n and iteration <= max_iterations:
                iteration += 1

                # REMARK: only the original is fed in twice instead of the template
                raw_counterfactual_tensor = self.model.predict(
                    originals_tensor, originals_tensor
                )
                raw_counterfactuals = self.reverse_transform(raw_counterfactual_tensor)
                counterfactuals = self._post_process_transformed_counterfactuals(
                    raw_counterfactuals, repeated_cf_templates
                )

                metrics = self.evaluate_counterfactuals(
                    counterfactuals, repeated_originals, classifier, ctgan, batch_size
                )["global_metrics"]

                # only gather the valid samples
                valid = metrics["valid_counterfactuals"]

                if valid.sum() != 0:
                    new_valid_counterfactuals = pd.concat(
                        [
                            counterfactuals[valid],
                            metrics[valid],
                        ],
                        axis=1,
                    )
                    valid_counterfactuals = pd.concat(
                        [
                            valid_counterfactuals,
                            new_valid_counterfactuals,
                        ],
                        axis=0,
                    )

                pbar.n = len(valid_counterfactuals)
                pbar.refresh()

        if len(valid_counterfactuals) == 0:
            print("No valid samples were generated. Try again!")
            return valid_counterfactuals
        if sort:
            valid_counterfactuals = valid_counterfactuals.sort_values(
                ["counterfactual_pred"], ascending=False
            )
        if len(valid_counterfactuals) > n:
            valid_counterfactuals = valid_counterfactuals[:n]

        print(
            f"Out of {iteration * batch_size} proposals "
            f"{len(valid_counterfactuals)}/{n} samples of good quality."
        )
        return valid_counterfactuals

    def _disc_step(self, originals: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Perform a discriminator step.

        Args:
            originals (torch.Tensor): Original samples.
            device (torch.device): Device to use for training.

        Returns:
            torch.Tensor: Discriminator loss.
        """
        batch_size = originals.shape[0]

        keep_cols = generate_counterfactual_masks(
            dropout=self.config["cf_dropout_range"],
            batch_size=batch_size,
            no_dropout_cols=[self.config["target_name"]],
            metadata=self.metadata,
        )
        keep_transformed = transform_mask(mask=keep_cols, metadata=self.metadata)
        masked_originals = apply_mask(tensor=originals, mask=keep_transformed)

        # randomly select other target value as counterfactual
        counterfactual_templates = randomly_change_to_other_cat(
            data=masked_originals,
            column=self.config["target_name"],
            metadata=self.metadata,
        )

        noise = torch.normal(
            mean=0, std=1, size=(batch_size, self.config["embedding_dim"])
        )
        # REMARK: only the original is fed in twice instead of the template
        generator_input = torch.cat([originals, originals, noise], dim=1)
        raw_counterfactuals = self.model.generator(generator_input)
        counterfactuals = self._post_process_transformed_counterfactuals(
            raw_counterfactuals, counterfactual_templates
        )

        # Sample real counterfactuals
        target_cat_col_id = self.metadata.column_to_idx(
            self.config["target_name"], only_categories=True
        )
        target_cat_col_ids = torch.full((batch_size,), target_cat_col_id)
        transformed_target_idxs = self.metadata.column_to_transformed_idxs(
            self.config["target_name"]
        )
        value_ids = counterfactuals[:, transformed_target_idxs].argmax(dim=1)
        real_counterfactuals = self.sampler.sample_data(
            cat_col_ids=target_cat_col_ids, value_ids=value_ids
        )

        # only original/counterfactual fed in -> real/fake
        disc_originals_pred = self.model.discriminator(originals).reshape(-1)
        disc_real_counterfactual_pred = self.model.discriminator(real_counterfactuals)
        disc_fake_pred = self.model.discriminator(counterfactuals).reshape(-1)

        gp = calc_gradient_penalty(
            real_data=originals,
            fake_data=counterfactuals,
            discriminator=self.model.discriminator,
            pac=self.config["pac"],
            device=device,
            gradient_penalty_influence=self.config["gradient_penalty_influence"],
        )

        original_disc_loss = -(
            torch.mean(disc_originals_pred) - torch.mean(disc_fake_pred)
        )
        counterfactual_disc_loss = -(
            torch.mean(disc_real_counterfactual_pred) - torch.mean(disc_fake_pred)
        )

        return (
            original_disc_loss * self.config["original_disc_influence"]
            + counterfactual_disc_loss * self.config["counterfactual_disc_influence"]
            + gp
        )

    def _disc_update(self, disc_loss: torch.Tensor) -> None:
        """
        Update the discriminator.

        Args:
            disc_loss (torch.Tensor): Discriminator loss.
        """
        self._disc_optimizer.zero_grad()
        disc_loss.backward()
        self._disc_optimizer.step()

    def _gen_step(
        self, originals: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform a generator step.

        Args:
            originals (torch.Tensor): Original samples.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Generator \
                losses.
        """
        batch_size = originals.shape[0]

        keep_cols = generate_counterfactual_masks(
            dropout=self.config["cf_dropout_range"],
            batch_size=batch_size,
            no_dropout_cols=[self.config["target_name"]],
            metadata=self.metadata,
        )
        to_change = ~keep_cols
        keep_transformed = transform_mask(mask=keep_cols, metadata=self.metadata)

        # randomly select other target value as counterfactual
        unmasked_counterfactual_templates = randomly_change_to_other_cat(
            data=originals,
            column=self.config["target_name"],
            metadata=self.metadata,
        )
        counterfactual_templates = apply_mask(
            tensor=unmasked_counterfactual_templates, mask=keep_transformed
        )

        noise = torch.normal(
            mean=0, std=1, size=(batch_size, self.config["embedding_dim"])
        )
        # REMARK: only the original is fed in twice instead of the template
        generator_input = torch.cat([originals, originals, noise], dim=1)
        raw_counterfactuals = self.model.generator(generator_input)
        counterfactuals = self._post_process_transformed_counterfactuals(
            raw_counterfactuals, counterfactual_templates
        )

        disc_output_loss = torch.mean(-self.model.discriminator(counterfactuals))

        # normalize by number of unmasked features (and batch size)
        reconstruction_loss = masked_divergence_loss(
            predicted=raw_counterfactuals,
            target=counterfactual_templates,
            col_mask=keep_cols,
            metadata=self.metadata,
            continous_float_influence=self.config["continuous_float_influence"],
        ).mean()

        # normalize by number of unmasked features (and batch size)
        counterfactual_loss = masked_divergence_loss(
            predicted=counterfactuals,
            target=originals,
            col_mask=to_change,
            metadata=self.metadata,
            continous_float_influence=self.config["continuous_float_influence"],
        ).mean()

        disc_influence = (
            self.config["original_disc_influence"]
            + self.config["counterfactual_disc_influence"]
        )

        # classifier loss
        if self.classifier:
            target_col_idxs = self.metadata.column_to_transformed_idxs(
                self.config["target_name"]
            )
            feature_col_idxs = list(
                set(range(counterfactuals.shape[1])) - set(target_col_idxs)
            )
            target = torch.argmax(counterfactuals[:, target_col_idxs], dim=1)
            counterfactual_features = counterfactuals[:, feature_col_idxs]
            target_pred = self.classifier(counterfactual_features)
            classifier_loss = F.cross_entropy(target_pred, target).mean()
        else:
            classifier_loss = torch.tensor(0)

        return (
            disc_output_loss * disc_influence,
            reconstruction_loss * self.config["reconstruction_divergence_influence"],
            counterfactual_loss * self.config["cf_divergence_influence"],
            classifier_loss * self.config["classifier_influence"],
        )


class SimpleNoTemplateFCEGAN(NoTemplateFCEGAN):
    """
    SimpleNoTemplateFCEGAN is a class that extends NoTemplateFCEGAN to provide
    a simple data transformation instead of the CTGAN data transformation on top of
    the ablation implemented by the NoTemplateFCEGAN. As mentioned before, this model
    has only illustrative purposes. FCEGAN and SimpleFCEGAN are the main models.

    Rewritten Methods:
        initialize_training
        load_state_dict
    """

    def initialize_training(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        target_name: str,
        classifier: Optional[nn.Module] = None,
        batch_size: int = 300,
        state_dict: Optional[Dict[str, Any]] = None,
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Initialize training with the given training and test data.

        Args:
            train (pd.DataFrame): Training data.
            test (pd.DataFrame): Test data.
            target_name (str): Target column name.
            classifier (Optional[nn.Module]): Pre-trained classifier.
            batch_size (int): Batch size.
            state_dict (Optional[Dict[str, Any]]): State dictionary for loading \
                model state.

        Returns:
            Tuple[DataLoader, DataLoader]: Data loaders for training and test data.
        """
        logger.info("Initializing for training.")

        self.config["target_name"] = target_name

        if classifier:
            self.classifier = classifier

        if state_dict:
            self.load_state_dict(state_dict)

        if batch_size > len(test) or batch_size > len(train):
            raise ValueError(f"Batch size of {batch_size} too big.")
        elif batch_size % self.config["pac"]:
            raise ValueError("Batch size has to be divisible by pac size")

        features = train.columns.to_list()
        continuous_cols = infer_continuous_columns(train)
        categorical_cols = list(set(features) - set(continuous_cols))

        if not self.transformer.metadata:
            self.transformer.fit(raw_data=train, categorical_columns=categorical_cols)
        self.metadata = adjust_metadata_for_simple_transform(self.transformer.metadata)

        self.transform = partial(simple_transform, metadata=self.metadata)
        self.reverse_transform = partial(
            reverse_simple_transform, metadata=self.metadata
        )

        train_transformed = self.transform(train)
        test_transformed = self.transform(test)

        train_dataset = OnlyFeaturesDataset(train_transformed)
        test_dataset = OnlyFeaturesDataset(test_transformed)

        train_dataloader = DataLoader(
            train_dataset, batch_size, shuffle=True, drop_last=True
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size, shuffle=True, drop_last=True
        )

        if not state_dict:
            train_tf = self.transform(train)
            self.sampler.fit(train_tf, self.metadata)
            self._init_model_and_optim()

        return train_dataloader, test_dataloader

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """
        Load the state dictionary.

        Args:
            state_dict (Dict[str, Any]): State dictionary to load.
        """
        self.config = state_dict["config"]
        self.epoch = self.config["epoch"]

        self.transformer = CtganTransformer()
        self.transformer.load_state_dict(state_dict["transformer_state_dict"])
        self.metadata = adjust_metadata_for_simple_transform(self.transformer.metadata)

        self.transform = partial(simple_transform, metadata=self.metadata)
        self.reverse_transform = partial(
            reverse_simple_transform, metadata=self.metadata
        )

        self.sampler = DataSampler()
        self.sampler.load_state_dict(state_dict["sampler_state_dict"])

        self._init_model_and_optim()
        self.model.load_state_dict(state_dict["model_state_dict"])
        self._gen_optimizer.load_state_dict(state_dict["gen_optimizer_state_dict"])
        self._disc_optimizer.load_state_dict(state_dict["disc_optimizer_state_dict"])


def evaluate_counterfactuals(
    counterfactuals: pd.DataFrame,
    originals: pd.DataFrame,
    classifier: nn.Module,
    classifier_transform: Callable[[pd.DataFrame], torch.Tensor],
    ctgan: CtganSynthesizer,
    metadata: MetaData,
    target_name: str,
    batch_size: int = 300,
):
    """
    Evaluate the generated counterfactuals.

    Args:
        counterfactuals (pd.DataFrame): Generated counterfactuals.
        originals (pd.DataFrame): Original samples.
        classifier (nn.Module): Classifier to validate counterfactuals.
        ctgan (CtganSynthesizer): Pre-trained CTGAN synthesizer.
        batch_size (int): Batch size.

    Returns:
        Dict[str, Any]: Evaluation metrics.
    """
    classifier.to(torch.device("cpu"))
    ctgan.to(torch.device("cpu"))

    # 1) classifier target shift
    cf_dataset = PandasDataset(
        counterfactuals,
        target_name,
        classifier_transform,
    )
    originals_set = PandasDataset(
        originals,
        target_name,
        classifier_transform,
    )

    cf_loader = DataLoader(cf_dataset, batch_size=batch_size)
    original_loader = DataLoader(originals_set, batch_size=batch_size)

    prediction_gains = []
    cf_predictions = []
    valid_counterfactuals = []

    classifier.eval()
    with torch.no_grad():
        for (cf_features, cf_targets), (og_features, _) in zip(
            cf_loader, original_loader
        ):
            cf_logits = classifier(cf_features)
            og_logits = classifier(og_features)

            cf_prediction = F.softmax(cf_logits, dim=1)[
                range(cf_features.shape[0]), cf_targets
            ]
            original_prediction = F.softmax(og_logits, dim=1)[
                range(cf_features.shape[0]), cf_targets
            ]

            prediction_gains.append(cf_prediction - original_prediction)
            cf_predictions.append(cf_prediction)

            # Check if the argmax is effectively the cf_targets
            cf_argmax = torch.argmax(cf_logits, dim=1)
            correct_predictions = (cf_argmax == cf_targets).float()
            valid_counterfactuals.append(correct_predictions)

        prediction_gains = torch.cat(prediction_gains, dim=0)
        cf_predictions = torch.cat(cf_predictions, dim=0)
        valid_counterfactuals = torch.cat(valid_counterfactuals, dim=0)

    # 2) divergence metrics
    unmasked_counterfactual_templates = originals.copy()
    unmasked_counterfactual_templates[target_name] = counterfactuals[target_name]
    distances = calculate_distances(
        unmasked_counterfactual_templates, counterfactuals, metadata
    )
    cat_changed = np.mean(distances["hamming"], axis=1)
    mean_percentile_shift = distances["percentile_shifts"].abs().mean(axis=1)
    max_percentile_shift = distances["percentile_shifts"].abs().max(axis=1)

    # 3) include realism - discriminator
    # Make conditional vectors
    col_id = ctgan.metadata.column_to_idx(target_name, only_categories=True)
    cond_vector_dim = ctgan.sampler.dim_conditional_vector()
    cond_vector_start = ctgan.sampler.categorical_column_cond_start_id[col_id]
    value_ids = torch.argmax(
        ctgan.transformer.transform(counterfactuals[target_name]), dim=1
    )
    cond_vecs = torch.zeros((len(value_ids), cond_vector_dim))
    cond_vecs[torch.arange(len(value_ids)), cond_vector_start + value_ids] = 1.0

    # TODO: make batches
    counterfactuals = ctgan.transformer.transform(counterfactuals)
    discriminator_input = torch.cat([counterfactuals, cond_vecs], dim=1)
    # Discriminator works per pacs
    discriminator_input_repeated = discriminator_input.repeat_interleave(
        ctgan.config["pac"], dim=0
    )
    with torch.no_grad():
        fakeness = ctgan.model.discriminator(discriminator_input_repeated).squeeze()

    global_metrics = pd.DataFrame(
        {
            "cat_changed": cat_changed,
            "mean_percentile_shift": mean_percentile_shift,
            "max_percentile_shift": max_percentile_shift,
            "counterfactual_pred": cf_predictions,
            "prediction_gain": prediction_gains,
            "valid_counterfactuals": valid_counterfactuals,
            "fakeness": fakeness,
        }
    )

    return {**distances, "global_metrics": global_metrics}


def post_process_transformed_counterfactuals(
    raw_counterfactuals: torch.Tensor,
    counterfactual_templates: torch.Tensor,
    metadata: MetaData,
):

    # make mask
    to_change_mask = []
    start = 0
    for col_meta in metadata:
        output_dim = col_meta.output_dimension
        end = start + output_dim
        to_change_col = counterfactual_templates[:, start:end].sum(dim=1) == 0
        to_change_mask.append(to_change_col.unsqueeze(1).repeat(1, output_dim))
        start = end

    to_change_mask = torch.cat(to_change_mask, dim=1)

    # make counterfactuals
    counterfactuals = counterfactual_templates + apply_mask(
        tensor=raw_counterfactuals, mask=to_change_mask
    )
    return counterfactuals


def post_process_counterfactuals(
    raw_counterfactuals: pd.DataFrame,
    counterfactual_templates: pd.DataFrame,
):
    """
    Post-process generated counterfactuals.

    Args:
        raw_counterfactuals (pd.DataFrame): Raw generated counterfactuals.
        counterfactual_templates (pd.DataFrame): Counterfactual templates.

    Returns:
        pd.DataFrame: Post-processed counterfactuals.
    """
    return counterfactual_templates.reset_index(drop=True).fillna(
        raw_counterfactuals.reset_index(drop=True)
    )
