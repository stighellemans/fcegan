"""
This module provides utilities for accumulating sums, managing model checkpoints,
and plotting metrics.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import torch

from .utils import remove_folder


class Accumulator:
    """For accumulating sums over `n` variables."""

    def __init__(self, n: int):
        """
        Initializes an Accumulator with `n` variables.

        Args:
            n (int): Number of variables to accumulate.
        """
        self.data = [0.0] * n

    def add(self, *args: float) -> None:
        """
        Adds values to the accumulator.

        Args:
            *args (float): Values to add to the accumulator.
        """
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self) -> None:
        """Resets the accumulator to zero."""
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx: int) -> float:
        """
        Gets the value at a specific index.

        Args:
            idx (int): Index of the value to retrieve.

        Returns:
            float: Value at the specified index.
        """
        return self.data[idx]

    def __truediv__(self, other: Union[int, float]) -> List[float]:
        """
        Divides each element in the accumulator by a given value.

        Args:
            other (Union[int, float]): The value to divide by.

        Returns:
            List[float]: The result of the division.

        Raises:
            TypeError: If the other value is not an int or float.
        """
        if isinstance(other, (int, float)):
            # Divide each element in the data list by the other value
            return [a / other for a in self.data]
        else:
            raise TypeError("Unsupported type for division")


class BookKeeper:
    """Handles saving and updating metrics and model checkpoints."""

    def __init__(
        self, save_path: Union[str, Path], best_metric_name: Optional[str] = None
    ):
        """
        Initializes the BookKeeper.

        Args:
            save_path (Union[str, Path]): Path to save checkpoints and metrics.
            best_metric_name (Optional[str]): Name of the metric to determine the \
                best epoch.
        """
        self.save_path = Path(save_path)
        self.checkpoint_base_path = self.save_path / "checkpoints"
        self.metrics_path = self.save_path / "metrics.json"

        self.metrics = load_metrics(self.metrics_path)
        self.best_metric_name = best_metric_name

        self.epoch, self.best_epoch, self.best_metric = 0, 0, -float("inf")

    @property
    def checkpoint_paths(self) -> List[Path]:
        """
        Returns the paths of all checkpoints in the checkpoint directory.

        Returns:
            List[Path]: List of checkpoint paths.
        """
        return get_checkpoint_paths(self.checkpoint_base_path)

    def load_checkpoint(
        self,
        resume: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Loads a checkpoint to resume training.

        Args:
            resume (bool, optional): Whether to resume from the last checkpoint. \
                Defaults to True.

        Returns:
            Optional[Dict[str, Any]]: The checkpoint data if available, otherwise None.

        Raises:
            ValueError: If there is a trained model already in the folder and \
                overwrite is not allowed.
        """
        if resume and self.checkpoint_base_path.exists():
            checkpoint_wrap = load_checkpoint(self.checkpoint_base_path)
            self.epoch = checkpoint_wrap["epoch"]
            self.remove_metrics(from_epoch=self.epoch + 1)
            return checkpoint_wrap["checkpoint"]
        elif self.checkpoint_base_path.exists():
            raise ValueError(
                "Trained model already in folder. This will not be overwritten."
                " Delete the content of the folder to continue."
            )
        else:
            return None

    def update(
        self,
        metrics_dict: Dict[str, Dict[str, Union[int, float]]],
    ) -> None:
        """
        Updates the metrics with new values.

        Args:
            metrics_dict (Dict[str, Dict[str, Union[int, float]]]): Dictionary \
                containing new metrics.
        """
        self.epoch += 1

        # Make new metrics dictionary
        if not self.metrics:
            self.metrics = {
                name: {metric_name: [] for metric_name in metrics.keys()}
                for name, metrics in metrics_dict.items()
            }

        # Update the metrics with the new values
        for name, metrics in metrics_dict.items():
            for metric_name, value in metrics.items():
                self.metrics[name][metric_name].append(value)

        plot_metrics(self.metrics, self.save_path)
        save_metrics(self.metrics, self.metrics_path)

        # Update best metric
        if not self.best_metric_name:
            self.best_metric = self.epoch
        else:
            for metrics in metrics_dict.values():
                metric = metrics.get(self.best_metric_name, None)
                if metric is not None and metric > self.best_metric:
                    self.best_metric = metric
                    self.best_epoch = self.epoch

    def remove_metrics(self, from_epoch: int) -> None:
        """
        Removes metrics from a specified epoch.

        Args:
            from_epoch (int): The epoch from which to remove metrics.
        """
        for name, metrics in self.metrics.items():
            for metric_name in metrics.keys():
                self.metrics[name][metric_name] = self.metrics[name][metric_name][
                    : from_epoch - 1
                ]

    def save_checkpoint(
        self,
        checkpoint: Dict[str, Any],
        saving_each_x_epoch: int = 1,
        only_best: bool = False,
    ) -> None:
        """
        Saves a checkpoint of the model.

        Args:
            checkpoint (Dict[str, Any]): The checkpoint data.
            saving_each_x_epoch (int, optional): Frequency of saving checkpoints. \
                Defaults to 1. `-1` always saves the last epoch.
            only_best (bool, optional): Whether to save only the best checkpoint. \
                Defaults to False.

        Raises:
            ValueError: If saving_each_x_epoch is an invalid value.
        """
        if saving_each_x_epoch == 0 or (only_best and self.epoch != self.best_epoch):
            return
        elif saving_each_x_epoch == -1 or (only_best and self.epoch == self.best_epoch):
            remove_folder(self.checkpoint_base_path)
        elif saving_each_x_epoch <= -2:
            raise ValueError(f"Saving each {saving_each_x_epoch} epoch not possible.")

        self.checkpoint_base_path.mkdir(parents=True, exist_ok=True)

        checkpoint_wrap = {"epoch": self.epoch, "checkpoint": checkpoint}
        save_checkpoint(
            checkpoint_wrap,
            checkpoint_base_path=self.checkpoint_base_path,
            checkpoint_name=f"model_epoch_{self.epoch}.pth",
        )


# # # # # # # # separate bookkeeping functions # # # # # # # #


def plot_metrics(
    metrics_dict: Dict[str, Dict[str, Any]],
    save_path: Optional[Union[Path, str]] = None,
) -> None:
    """
    Plots metrics and saves or shows the plot.

    Args:
        metrics_dict (Dict[str, Dict[str, Any]]): Dictionary containing metrics to plot.
        save_path (Optional[Union[Path, str]], optional): Path to save the plot. \
            Defaults to None.
    """
    num_subplots = len(metrics_dict)
    plt.figure(figsize=(12, 4 * num_subplots))

    # Iterate through each subplot data
    for i, (subplot_title, metrics) in enumerate(metrics_dict.items(), 1):
        epochs = range(1, len(metrics[list(metrics)[0]]) + 1)

        plt.subplot(num_subplots, 1, i)
        for label, data in metrics.items():
            plt.plot(epochs, data, label=label)
        plt.xlabel("Epochs")
        plt.ylabel("Metrics")
        plt.title(subplot_title)
        plt.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(Path(save_path) / "metrics.png")
        plt.close()
    else:
        plt.show()


def load_metrics(metrics_path: Path) -> Optional[Dict[str, Any]]:
    """
    Loads metrics from a JSON file.

    Args:
        metrics_path (Path): Path to the metrics JSON file.

    Returns:
        Optional[Dict[str, Any]]: Dictionary containing metrics data, or None if the \
            file doesn't exist.
    """
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        return metrics
    else:
        return None


def save_metrics(metrics: Dict[str, Any], metrics_path: Path) -> None:
    """
    Saves metrics to a JSON file.

    Args:
        metrics (Dict[str, Any]): Dictionary containing metrics data.
        metrics_path (Path): Path to the metrics JSON file.
    """
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)


def save_checkpoint(
    checkpoint: Dict[str, Any],
    checkpoint_base_path: Union[str, Path],
    checkpoint_name: str,
) -> None:
    """
    Saves a model checkpoint.

    Args:
        checkpoint (Dict[str, Any]): The checkpoint data.
        checkpoint_base_path (Union[str, Path]): Path to the directory where the \
            checkpoint will be saved.
        checkpoint_name (str): Name of the checkpoint file.
    """
    torch.save(checkpoint, Path(checkpoint_base_path) / checkpoint_name)


def load_checkpoint(
    checkpoint_base_path: Union[str, Path],
) -> Optional[Dict[str, Any]]:
    """
    Loads the latest checkpoint from the checkpoint directory.

    Args:
        checkpoint_base_path (Union[str, Path]): Path to the directory containing \
            checkpoints.

    Returns:
        Optional[Dict[str, Any]]: The latest checkpoint data, or None if no \
            checkpoints are found.
    """
    checkpoints = get_checkpoint_paths(checkpoint_base_path)
    if checkpoints:
        latest_checkpoint = checkpoints[-1]
        return torch.load(latest_checkpoint)
    else:
        return None


def get_checkpoint_paths(
    checkpoint_base_path: Union[str, Path], sort_paths: bool = True
) -> List[Path]:
    """
    Retrieves the paths of all checkpoint files in the given directory.

    Args:
        checkpoint_base_path (Union[str, Path]): Path to the directory where the \
            checkpoints are stored.
        sort_paths (bool, optional): Whether to sort the checkpoint paths. Defaults \
            to True.

    Returns:
        List[Path]: List of checkpoint paths.

    Raises:
        ValueError: If checkpoint filenames do not follow the expected format.
    """
    checkpoint_base_path = Path(checkpoint_base_path)
    pt_paths = list(checkpoint_base_path.glob("*.pt"))
    pth_paths = list(checkpoint_base_path.glob("*.pth"))
    all_paths = pt_paths + pth_paths

    if sort_paths:
        try:
            return sorted(all_paths, key=lambda x: int(x.stem.split("_")[-1]))
        except ValueError as e:
            raise ValueError(
                "Filename format not correct. Expected: '*_(int).[pt|pth]'"
            ) from e
    else:
        return all_paths
