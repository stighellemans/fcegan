"""
This module contains the implementation of the CounterfactualOptimizer class, which
optimizes counterfactual samples to achieve desired outcomes while balancing multiple
loss components such as classifier loss, divergence loss, and realism loss.
"""

from typing import Callable, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from ..data.metadata import MetaData
from ..data.utils import cf_template_to_keep_mask, transform_mask
from ..utils.bookkeeping import BookKeeper
from ..utils.utils import move_optimizer_to_device
from .counterfactual_synthesizer import (
    evaluate_counterfactuals,
    post_process_counterfactuals,
)
from .ctgan_synthesizer import CtganSynthesizer
from .losses import RealismLoss, masked_divergence_loss


class CounterfactualOptimizer:
    """
    Optimizes counterfactual samples to achieve desired outcomes using a combination of
    classifier loss, divergence loss, and realism loss.

    Attributes:
        classifier (nn.Module): The classifier model.
        ctgan (CtganSynthesizer): The CTGAN synthesizer.
        transformer (CtganTransformer): Transformer for data preprocessing.
        device (torch.device): Device to perform computations on.
        lr (float): Learning rate for the optimizer.
        betas (Tuple[float, float]): Betas for the Adam optimizer.
        classifier_influence (float): Weight of the classifier loss.
        divergence_influence (float): Weight of the divergence loss.
        realism_influence (float): Weight of the realism loss.
    """

    def __init__(
        self,
        classifier: nn.Module,
        train_ctgan: CtganSynthesizer,
        transform_fn: Callable[[pd.DataFrame], torch.Tensor],
        reverse_transform_fn: Callable[[torch.Tensor], pd.DataFrame],
        metadata: MetaData,
        lr: float = 1e-1,
        betas: Tuple[float, float] = (0.9, 0.999),
        classifier_influence: float = 1,
        divergence_influence: float = 10,
        realism_influence: float = 0.1,
        **kwargs,
    ) -> None:
        """
        Initialize the optimization engine.

        Args:
            classifier (nn.Module): The classifier model to be used.
            train_ctgan (CtganSynthesizer): The CTGAN synthesizer for training.
            transform_fn (Callable[[pd.DataFrame], torch.Tensor]): Function to\
                  transform a DataFrame to a tensor.
            reverse_transform_fn (Callable[[torch.Tensor], pd.DataFrame]): Function to\
                  reverse transform a tensor to a DataFrame.
            metadata (MetaData): Metadata information.
            lr (float, optional): Learning rate for the optimizer. Defaults to 1e-1.
            betas (Tuple[float, float], optional): Coefficients used for computing\
                  running averages of gradient and its square. Defaults to (0.9, 0.999).
            classifier_influence (float, optional): Influence of the classifier in the\
                  optimization process. Defaults to 1.
            divergence_influence (float, optional): Influence of the divergence in the\
                  optimization process. Defaults to 10.
            realism_influence (float, optional): Influence of the realism in the\
                  optimization process. Defaults to 0.1.
        """

        self.classifier = classifier
        self.train_ctgan = train_ctgan
        self.transform = transform_fn
        self.reverse_transform = reverse_transform_fn
        self.metadata = metadata
        self.device = torch.device("cpu")

        self.lr = lr
        self.betas = betas
        self.classifier_influence = classifier_influence
        self.divergence_influence = divergence_influence
        self.realism_influence = realism_influence

    def to(self, device: torch.device) -> None:
        """
        Moves the optimizer to the specified device.

        Args:
            device (torch.device): The device to move the optimizer to.
        """
        self.device = device

    def optimize_counterfactuals(
        self,
        samples: pd.DataFrame,
        cf_templates: pd.DataFrame,
        target_name: str,
        num_steps: int = 20,
        template_guided: bool = True,
        evaluate_each: int = -1,
        verbose: bool = True,
        save_path: str = "",
    ) -> pd.DataFrame:
        """
        Optimizes counterfactuals over a number of steps and returns the final
        counterfactuals.

        Args:
            samples (pd.DataFrame): The original samples.
            cf_templates (pd.DataFrame): Templates for counterfactuals.
            target_name (str): The name of the target variable.
            num_steps (int, optional): Number of optimization steps. Defaults to 20.
            evaluate_each (int, optional): Evaluate every 'evaluate_each' steps.
                Defaults to -1.
            verbose (bool, optional): Whether to display progress. Defaults to True.
            save_path (str, optional): Path to save evaluation metrics. Defaults to "".

        Returns:
            pd.DataFrame: The optimized counterfactuals.
        """
        if evaluate_each == -1:
            evaluate_each = num_steps
        elif evaluate_each == 0 or evaluate_each < -1:
            evaluate_each = None

        if save_path:
            bookkeeper = BookKeeper(save_path=save_path)

        self._initialize_training(target_name, samples, cf_templates)

        pbar = tqdm(range(1, num_steps + 1), disable=not verbose)
        for step in pbar:
            target_logits = self.classifier(self.features)
            counterfactuals_tf = torch.concat(
                [self.features, self.cf_targets_tensor], dim=1
            ).to(self.device)
            classifier_loss = (
                self.classifier_influence
                * F.cross_entropy(target_logits, self.cf_targets).mean()
            )
            divergence_loss = self.divergence_influence * masked_divergence_loss(
                counterfactuals_tf,
                self.base_counterfactuals,
                col_mask=self.keep_features,
                metadata=self.metadata,
            ).mean().to(self.device)

            if self.realism_influence > 0:
                fakeness = (
                    self.realism_influence
                    * self.realism_loss_fn(counterfactuals_tf).to(self.device).mean()
                )
            else:
                fakeness = 0

            loss = classifier_loss + divergence_loss + fakeness

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if template_guided:
                # Keep features specified as immutable in the template
                with torch.no_grad():
                    self.features[self.keep_features_tf] = self.old_features[
                        self.keep_features_tf
                    ]

            # Evaluation
            if evaluate_each is not None and (step % evaluate_each == 0):
                raw_counterfactuals_tf = (
                    torch.concat([self.features, self.cf_targets_tensor], dim=1)
                    .detach()
                    .cpu()
                )
                raw_counterfactuals = self.reverse_transform(raw_counterfactuals_tf)
                counterfactuals = post_process_counterfactuals(
                    raw_counterfactuals, cf_templates
                )

                self.classifier.to(torch.device("cpu"))
                self.train_ctgan.to(torch.device("cpu"))
                mean_metrics = (
                    evaluate_counterfactuals(
                        counterfactuals,
                        samples,
                        self.classifier,
                        self.transform,
                        self.train_ctgan,
                        self.metadata,
                        target_name,
                        batch_size=len(samples) // 10,
                    )["global_metrics"]
                    .mean()
                    .to_dict()
                )

                if save_path:
                    bookkeeper.update({"Quality measures": mean_metrics})

                pbar.set_postfix(
                    {
                        "pred": mean_metrics["counterfactual_pred"],
                        "cat_changed": mean_metrics["cat_changed"],
                        "mean_perc": mean_metrics["mean_percentile_shift"],
                        "max_perc": mean_metrics["max_percentile_shift"],
                        "fake": mean_metrics["fakeness"],
                    }
                )
                self.classifier.to(self.device)
                self.train_ctgan.to(self.device)

            raw_counterfactuals_tf = (
                torch.concat([self.features, self.cf_targets_tensor], dim=1)
                .detach()
                .cpu()
            )
            raw_counterfactuals = self.reverse_transform(raw_counterfactuals_tf)
            counterfactuals = post_process_counterfactuals(
                raw_counterfactuals, cf_templates
            )

        return counterfactuals

    def _initialize_training(
        self, target_name: str, samples: pd.DataFrame, cf_templates: pd.DataFrame
    ) -> None:
        """
        Initializes training by setting up all required tensors and variables.

        Args:
            target_name (str): The name of the target variable.
            samples (pd.DataFrame): The original samples.
            cf_templates (pd.DataFrame): Templates for counterfactuals.
        """
        self.target_name = target_name
        target_idx = [
            i
            for i, column_meta in enumerate(self.metadata)
            if column_meta.column_name != target_name
        ]
        tf_target_idxs = self.metadata.column_to_transformed_idxs(target_name)
        all_tf_col_indices = np.arange(self.metadata.num_transformed_columns())

        self.feature_idxs = [
            i for i in range(self.metadata.num_columns()) if i != target_idx
        ]
        self.tf_feature_idxs = list(set(all_tf_col_indices) - set(tf_target_idxs))

        self.metadata = self.metadata

        # translate templates to usable information
        to_change_mask = transform_mask(
            torch.tensor(cf_templates.isna().to_numpy(), dtype=torch.bool),
            self.metadata,
        )

        samples_tf = self.transform(samples)
        features = samples_tf[:, self.tf_feature_idxs]
        cf_templates_tensor = samples_tf.masked_fill(to_change_mask, 0)

        # change to right counterfactual target
        target_idxs = self.metadata.column_to_transformed_idxs(target_name)
        cf_targets = self.transform(cf_templates[target_name])
        cf_templates_tensor[:, target_idxs] = cf_targets

        # all requirements for optimization
        self.cf_templates_tensor = cf_templates_tensor.to(self.device)

        keep_cols = cf_template_to_keep_mask(
            cf_templates_tensor, self.metadata, transformed=False
        ).to(self.device)
        self.keep_features = keep_cols[:, self.feature_idxs].to(self.device)
        self.keep_features_tf = transform_mask(mask=keep_cols, metadata=self.metadata)[
            :, self.tf_feature_idxs
        ].to(self.device)

        self.features = features.to(self.device).requires_grad_(True)
        self.cf_targets_tensor = cf_templates_tensor[:, target_idxs].to(self.device)
        self.cf_targets = self.cf_targets_tensor.argmax(dim=1).to(self.device)

        self.old_features = self.features.clone().to(self.device)
        self.base_counterfactuals = torch.concat(
            [self.old_features, self.cf_targets_tensor], dim=1
        ).to(self.device)

        self.optimizer = optim.Adam([self.features], lr=self.lr, betas=self.betas)

        move_optimizer_to_device(self.optimizer, self.device)

        if self.realism_influence > 0:
            self.realism_loss_fn = RealismLoss(self.train_ctgan, target_name).to(
                self.device
            )
        self.classifier.to(self.device)
