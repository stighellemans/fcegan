"""
This module provides training functions for various models, including a standard
classifierand several types of FCEGANs (Counterfactual Generative Adversarial Networks).
These functions handle data loading, model initialization, training loop, evaluation,
and checkpointing.
"""

from pathlib import Path
from pprint import pprint
from typing import Any, Callable, Dict, Literal, Tuple, Union

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.datasets import PandasDataset
from ..data.metadata import MetaData
from ..data.transformer import CtganTransformer, SimpleTransformer
from ..data.utils import (
    calculate_diversity,
    generate_counterfactual_templates,
    transform_mask,
)
from ..models.classifiers import Classifier
from ..utils.bookkeeping import BookKeeper
from ..utils.utils import (
    check_available_device,
    load_json_file,
    move_optimizer_to_device,
    proceed_if_filedir_exists,
    same_config,
    write_json_file,
)
from .counterfactual_synthesizer import (
    FCEGAN,
    NoTemplateFCEGAN,
    SimpleFCEGAN,
    SimpleNoTemplateFCEGAN,
)
from .ctgan_synthesizer import CtganSynthesizer, SimpleCtganSynthesizer
from .losses import FocalLoss
from .optimization import CounterfactualOptimizer


def train_classifier(
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: Dict[str, Any],
    save_path: Union[str, Path],
    metadata: MetaData,
    transform: Callable,
    resume: bool = True,
    saving_each_x_epoch: int = 1,
    only_best: bool = False,
) -> None:
    """
    Trains a classifier on the given training and test datasets.

    Args:
        train (pd.DataFrame): Training dataset.
        test (pd.DataFrame): Test dataset.
        config (Dict[str, Any]): Configuration dictionary.
        save_path (Union[str, Path]): Path to save the model and checkpoints.
        metadata (MetaData): Metadata associated with the dataset.
        transform (Callable): Function to transform the data.
        resume (bool, optional): Whether to resume training from the last checkpoint. \
            Defaults to True.
        saving_each_x_epoch (int, optional): Save checkpoint after every x epochs. \
            Defaults to 1. `-1` saves only the last epoch.
        only_best (bool, optional): Save only the best model based on evaluation \
            metric. Defaults to False.

    Returns:
        None
    """

    # data
    train_set = PandasDataset(train, config["target_name"], transform)
    test_set = PandasDataset(test, config["target_name"], transform)

    train_loader = DataLoader(train_set, batch_size=config["batch_size"])
    test_loader = DataLoader(test_set, batch_size=config["batch_size"])

    # model
    config["input_dim"] = train_set[0][0].shape[0]
    config["dropout"] = config.get("dropout", 0)

    classifier = Classifier(**config)

    device = check_available_device()

    # criterion
    config["loss"] = loss_type = config.get("loss", "default")
    class_weights = [
        1 / v for v in metadata[config["target_name"]].value_frequencies.values()
    ]
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    if loss_type == "weighted":
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    elif loss_type == "focal":
        criterion = FocalLoss(weights=class_weights_tensor)
    elif loss_type == "default":
        criterion = nn.CrossEntropyLoss()

    # optimizer
    lr = config["lr"]
    config["weight_decay"] = config.get("weight_decay", 0)
    optimizer = optim.Adam(
        classifier.parameters(), lr=lr, weight_decay=config["weight_decay"]
    )

    print("Training started with configuration:")
    if not proceed_and_write_config(config, save_path, resume):
        return
    pprint(config)

    # checkpointing
    bookkeeper = BookKeeper(save_path, best_metric_name="Test_accuracy")
    checkpoint = bookkeeper.load_checkpoint(resume)
    if checkpoint:
        classifier.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    move_optimizer_to_device(optimizer, device)
    classifier.to(device)

    # training
    num_epochs = config["epochs"]
    for epoch in range(bookkeeper.epoch + 1, num_epochs + 1):
        train_loss = train_step(classifier, train_loader, optimizer, criterion, device)
        test_loss, test_accuracy = evaluate(classifier, test_loader, criterion, device)

        bookkeeper.update(
            {
                "Losses": {"Train_loss": train_loss, "Test_loss": test_loss},
                "Accuracy": {"Test_accuracy": test_accuracy},
            }
        )
        bookkeeper.save_checkpoint(
            checkpoint={
                "model_state_dict": classifier.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            saving_each_x_epoch=saving_each_x_epoch,
            only_best=only_best,
        )

        print(
            f"Epoch {epoch}/{num_epochs}, Train Loss: {train_loss:.4f}, "
            f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}"
        )


def train_ctgan(
    train_data: pd.DataFrame,
    data_transformer: CtganTransformer,
    epochs: int,
    save_path: Union[str, Path],
    simple_transform: bool = False,
    **kwargs,
):
    """
    Train a CTGAN (Conditional Tabular GAN) model using the provided training data and
    save the trained model.

    Args:
        train_data (pd.DataFrame): The training data to be used for training the CTGAN.
        data_transformer (CtganTransformer): The data transformer to be used for
            transforming the data.
        epochs (int): The number of epochs to train the CTGAN model.
        save_path (Union[str, Path]): The path where the trained CTGAN model
            will be saved. '.pth' extension needs to be included in the path.
        **kwargs: Additional keyword arguments to be passed to the CTGAN synthesizer.

    Returns:
        None
    """

    cat_cols = [
        column_meta.column_name
        for column_meta in data_transformer.metadata
        if column_meta.column_type == "categorical"
    ]

    if simple_transform:
        data_transformer = SimpleTransformer(data_transformer.metadata)
        ctgan = SimpleCtganSynthesizer(**kwargs)
    else:
        ctgan = CtganSynthesizer(**kwargs)

    # Train the CTGAN
    ctgan.transformer = data_transformer
    ctgan.initialize_training(train_data, cat_cols)

    for _ in tqdm(range(epochs)):
        ctgan.fit_epoch()

    torch.save(ctgan.state_dict(), save_path)


def train_fcegan(
    train: pd.DataFrame,
    test: pd.DataFrame,
    classifier: nn.Module,
    config: Dict[str, Any],
    save_path: Union[str, Path],
    fcegan_type: Literal[
        "fcegan", "no_template_fcegan", "simple_fcegan", "simple_no_template_fcegan"
    ] = "fcegan",
    resume: bool = True,
    saving_each_x_epoch: int = 1,
    only_best: bool = False,
) -> None:
    """
    Trains a FCEGAN with a given classifier and ctgan.

    Args:
        train (pd.DataFrame): Training dataset.
        test (pd.DataFrame): Test dataset.
        classifier (nn.Module): Pre-trained classifier model.
        eval_ctgan (CtganSynthesizer): Pre-trained CTGAN synthesizer to evaluate.
        config (Dict[str, Any]): Configuration dictionary.
        save_path (Union[str, Path]): Path to save the model and checkpoints.
        resume (bool, optional): Whether to resume training from the last checkpoint. \
            Defaults to True.
        saving_each_x_epoch (int, optional): Save checkpoint after every x epochs. \
            Defaults to 1. `-1` saves only the last epoch.
        only_best (bool, optional): Save only the best model based on evaluation \
            metric. Defaults to False.

    Returns:
        None
    """
    # synthesizer
    config["embedding_dim"] = config.get("embedding_dim", 128)
    config["discriminator_dim"] = config.get("discriminator_dim", (256, 256))
    config["generator_dim"] = config.get("generator_dim", (256, 256))
    config["discriminator_steps"] = config.get("discriminator_steps", 1)
    config["pac"] = config.get("pac", 10)
    config["cf_dropout_range"] = config.get("cf_dropout_range", (0.5, 1.0))
    config["discriminator_lr"] = config.get("discriminator_lr", 2e-4)
    config["generator_lr"] = config.get("generator_lr", 2e-4)
    config["discriminator_decay"] = config.get("discriminator_decay", 1e-6)
    config["generator_decay"] = config.get("generator_decay", 1e-6)
    config["betas"] = config.get("betas", (0.5, 0.9))
    config["gradient_penalty_influence"] = config.get("gradient_penalty_influence", 10)
    config["continuous_float_influence"] = config.get("continuous_float_influence", 1)
    config["counterfactual_disc_influence"] = config.get(
        "counterfactual_disc_influence", 0.5
    )
    config["original_disc_influence"] = config.get("original_disc_influence", 0.5)
    config["divergence_influence"] = config.get("divergence_influence", 1)
    config["batch_size"] = config.get("batch_size", 300)

    if fcegan_type == "fcegan":
        fcegan = FCEGAN(**config)
    elif fcegan_type == "no_template_fcegan":
        fcegan = NoTemplateFCEGAN(**config)
    elif fcegan_type == "simple_fcegan":
        fcegan = SimpleFCEGAN(**config)
    elif fcegan_type == "simple_no_template_fcegan":
        fcegan = SimpleNoTemplateFCEGAN(**config)
    else:
        raise ValueError(f"Invalid type of fcegan model: {fcegan_type}")

    # Load transformer if available
    config["transformer_path"] = config.get("transformer_path", None)
    if config["transformer_path"]:
        fcegan.transformer.load_state_dict(torch.load(config["transformer_path"]))

    # Load Data
    train_loader, test_loader = fcegan.initialize_training(
        train=train,
        test=test,
        classifier=classifier,
        target_name=config["target_name"],
        batch_size=config["batch_size"],
    )

    # Load CTGAN
    if "eval_ctgan_path" not in config:
        raise ValueError(
            "The configuration dictionary must contain a 'eval_ctgan_path' key."
        )
    config["eval_ctgan_path"] = config["eval_ctgan_path"]
    eval_ctgan = CtganSynthesizer()
    eval_ctgan.load_state_dict(torch.load(config["eval_ctgan_path"]))

    # device (not optimized for GPU)
    config["device"] = "cpu"

    print("Training started with configuration:")
    if not proceed_and_write_config(config, save_path, resume):
        return
    pprint(config)

    # checkpointing
    bookkeeper = BookKeeper(save_path, best_metric_name="valid_counterfactuals")
    checkpoint = bookkeeper.load_checkpoint(resume)
    if checkpoint:
        fcegan.load_state_dict(checkpoint)

    fcegan.to(config["device"])

    # training
    num_epochs = config["epochs"]
    pbar = tqdm(range(bookkeeper.epoch + 1, num_epochs + 1))
    for epoch in pbar:
        train_metrics = fcegan.fit_epoch(train_loader, test_loader, config["device"])
        test_metrics = (
            evaluate_fcegan(fcegan, test.sample(1000), classifier, eval_ctgan)[
                "global_metrics"
            ]
            .mean()
            .to_dict()
        )

        # Calculate diversity for a couple of random samples
        diversity_nums = 10
        diversity_metrics = []
        for _ in range(diversity_nums):
            i = torch.randint(0, len(test), (1,)).item()
            diversity_metrics.append(
                calc_fcegan_diversity(fcegan, test.iloc[i], batch_size=30)
            )

        mean_diversity = pd.DataFrame(diversity_metrics).mean()

        bookkeeper.update(
            {
                "Main Losses": {
                    "train_disc_loss": train_metrics["train_disc_loss"],
                    "train_gen_loss": train_metrics["train_gen_loss"],
                    "test_disc_loss": train_metrics["test_disc_loss"],
                    "test_gen_loss": train_metrics["test_gen_loss"],
                },
                "Generator Losses": {
                    "train_gen_by_disc_loss": train_metrics["train_gen_by_disc_loss"],
                    "train_reconst_divergence_loss": train_metrics[
                        "train_reconst_divergence_loss"
                    ],
                    "train_cf_divergence_loss": train_metrics[
                        "train_cf_divergence_loss"
                    ],
                    "train_classifier_loss": train_metrics["train_classifier_loss"],
                    "test_gen_by_disc_loss": train_metrics["test_gen_by_disc_loss"],
                    "test_reconst_divergence_loss": train_metrics[
                        "test_reconst_divergence_loss"
                    ],
                    "test_cf_divergence_loss": train_metrics["test_cf_divergence_loss"],
                    "test_classifier_loss": train_metrics["test_classifier_loss"],
                },
                "Quality Measures": {
                    "cat_changed": test_metrics["cat_changed"],
                    "mean_percentile_shift": test_metrics["mean_percentile_shift"],
                    "max_percentile_shift": test_metrics["max_percentile_shift"],
                    "counterfactual_prediction": test_metrics["counterfactual_pred"],
                    "prediction_gain": test_metrics["prediction_gain"],
                    "valid_counterfactuals": test_metrics["valid_counterfactuals"],
                    "fakeness": test_metrics["fakeness"],
                    "categorical_diversity": mean_diversity["categorical_diversity"],
                    "continuous_diversity": mean_diversity["continuous_diversity"],
                },
            }
        )

        bookkeeper.save_checkpoint(
            checkpoint=fcegan.state_dict(),
            saving_each_x_epoch=saving_each_x_epoch,
            only_best=only_best,
        )

        train_disc_loss = train_metrics["train_disc_loss"]
        train_gen_loss = train_metrics["train_gen_loss"]
        pbar.set_postfix(
            {
                "Epoch": epoch,
                "Train Disc Loss": train_disc_loss,
                "Train Gen Loss": train_gen_loss,
                "Test Valid Fraction": test_metrics["valid_counterfactuals"],
            }
        )


def evaluate_fcegan(
    fcegan: FCEGAN,
    test: pd.DataFrame,
    classifier: nn.Module,
    ctgan: CtganSynthesizer,
    batch_size: int = 300,
    dropout: float = 0.5,
):
    """
    Evaluates the FCEGAN using sampled data.

    Args:
        fcegan (FCEGAN): Trained fcegan synthesizer.
        test (pd.DataFrame): Test dataset.
        classifier (nn.Module): Pre-trained classifier model.
        ctgan (CtganSynthesizer): Pre-trained CTGAN synthesizer.
        batch_size (int, optional): Batch size for evaluation. Defaults to 300.
        dropout (float, optional): Dropout rate for counterfactual generation. \
            These are the features that are set mutable. Defaults to 0.5.

    Returns:
        Dict[str, Any]: Evaluation metrics.
    """
    # NOTE: at this time prediction is done in 1 time but danger of 'out of memory'
    target_name = fcegan.config["target_name"]
    counterfactual_templates = generate_counterfactual_templates(
        test, target_name, fcegan.metadata, dropout=dropout
    )
    target_idxs = fcegan.metadata.column_to_transformed_idxs(target_name)
    cf_targets = fcegan.transform(counterfactual_templates[target_name])
    to_change_mask = transform_mask(
        torch.tensor(counterfactual_templates.isna().to_numpy(), dtype=torch.bool),
        fcegan.metadata,
    )
    originals_tensor = fcegan.transform(test)
    cf_templates_tensor = originals_tensor.masked_fill(to_change_mask, 0)
    cf_templates_tensor[:, target_idxs] = cf_targets

    if isinstance(fcegan, NoTemplateFCEGAN):
        # REMARK: original tensor is fed in twice instead of the counterfactual template
        # This is to ensure comparibility between models (same amount of parameters)
        raw_counterfactual_tensor = fcegan.model.predict(
            originals_tensor, originals_tensor
        )
    else:
        raw_counterfactual_tensor = fcegan.model.predict(
            originals_tensor, cf_templates_tensor
        )
    raw_counterfactuals = fcegan.reverse_transform(raw_counterfactual_tensor)
    counterfactuals = fcegan.post_process_counterfactuals(
        raw_counterfactuals,
        counterfactual_templates,
    )

    metrics = fcegan.evaluate_counterfactuals(
        counterfactuals, test, classifier, ctgan, batch_size
    )

    return metrics


def calc_fcegan_diversity(
    fcegan: FCEGAN,
    sample: pd.Series,
    dropout: float = 0.5,
    batch_size: int = 30,
):
    """
    Calculates diversity metrics for a FCEGAN.

    Args:
        fcegan (FCEGAN): Trained FCEGAN synthesizer.
        sample (pd.Series): Sample data for diversity calculation.
        dropout (float, optional): Dropout rate for counterfactual generation. \
            These are the features that are set mutable. Defaults to 0.5.
        batch_size (int, optional): Batch size for diversity calculation. \
            Defaults to 30.

    Returns:
        Dict[str, float]: Diversity metrics.
    """
    target_name = fcegan.config["target_name"]
    sample = sample.to_frame().T
    counterfactual_template = generate_counterfactual_templates(
        sample, target_name, fcegan.metadata, dropout=dropout
    )
    repeated_sample = pd.concat([sample] * batch_size, axis=0).reset_index(drop=True)
    repeated_cf_template = pd.concat(
        [counterfactual_template] * batch_size, axis=0
    ).reset_index(drop=True)
    target_idxs = fcegan.metadata.column_to_transformed_idxs(target_name)
    cf_targets = fcegan.transform(repeated_cf_template[target_name])
    corruption_mask = transform_mask(
        torch.tensor(repeated_cf_template.isna().to_numpy(), dtype=torch.bool),
        fcegan.metadata,
    )
    originals_tensor = fcegan.transform(repeated_sample)
    cf_templates_tensor = originals_tensor.masked_fill(corruption_mask, 0)
    cf_templates_tensor[:, target_idxs] = cf_targets

    if isinstance(fcegan, NoTemplateFCEGAN):
        # REMARK: original tensor is fed in twice instead of the counterfactual template
        # This is to ensure comparibility between models (same amount of parameters)
        raw_counterfactual_tensor = fcegan.model.predict(
            originals_tensor, originals_tensor
        )
    else:
        raw_counterfactual_tensor = fcegan.model.predict(
            originals_tensor, cf_templates_tensor
        )
    raw_counterfactuals = fcegan.reverse_transform(raw_counterfactual_tensor)
    counterfactuals = fcegan.post_process_counterfactuals(
        raw_counterfactuals, repeated_cf_template
    )

    diversity = calculate_diversity(counterfactuals, metadata=fcegan.metadata)

    return {
        "categorical_diversity": diversity["hamming_mean"].mean(),
        "continuous_diversity": diversity["shifts_mean"].mean(),
    }


def calc_fcegan_diversity_random_generator(
    fcegan: FCEGAN,
    sample: pd.Series,
    dropout: float = 0.5,
    batch_size: int = 30,
):
    """
    Calculates diversity metrics for a random generator. The FCEGAN is purely used for
    its data transformations, methods and metadata.

    Args:
        fcegan FCEGAN: Trained fcegan synthesizer.
        sample (pd.Series): Sample data for diversity calculation.
        dropout (float, optional): Dropout rate for counterfactual generation. \
            These are the features that are set mutable. Defaults to 0.5.
        batch_size (int, optional): Batch size for diversity calculation. \
            Defaults to 30.

    Returns:
        Dict[str, float]: Diversity metrics.
    """
    target_name = fcegan.config["target_name"]
    sample = sample.to_frame().T
    counterfactual_template = generate_counterfactual_templates(
        sample, target_name, fcegan.metadata, dropout=dropout
    )
    repeated_sample = pd.concat([sample] * batch_size, axis=0).reset_index(drop=True)
    repeated_cf_template = pd.concat(
        [counterfactual_template] * batch_size, axis=0
    ).reset_index(drop=True)
    target_idxs = fcegan.metadata.column_to_transformed_idxs(target_name)
    cf_targets = fcegan.transform(repeated_cf_template[target_name])
    corruption_mask = transform_mask(
        torch.tensor(repeated_cf_template.isna().to_numpy(), dtype=torch.bool),
        fcegan.metadata,
    )
    originals_tensor = fcegan.transform(repeated_sample)
    cf_templates_tensor = originals_tensor.masked_fill(corruption_mask, 0)
    cf_templates_tensor[:, target_idxs] = cf_targets

    # REMARK: Random generator
    random_tensor = torch.rand_like(originals_tensor)

    raw_counterfactuals = fcegan.reverse_transform(random_tensor)
    counterfactuals = fcegan.post_process_counterfactuals(
        raw_counterfactuals, repeated_cf_template
    )

    diversity = calculate_diversity(counterfactuals, metadata=fcegan.metadata)

    return {
        "categorical_diversity": diversity["hamming_mean"].mean(),
        "continuous_diversity": diversity["shifts_mean"].mean(),
    }


def calc_optimization_diversity(
    cf_optimizer: CounterfactualOptimizer,
    sample: pd.Series,
    dropout: float = 0.5,
    num_steps: int = 20,
    batch_size: int = 30,
):
    """
    Calculates diversity metrics for an optimization-based counterfactual generator.

    Args:
        cf_optimizer (CounterfactualOptimizer): Counterfactual optimizer.
        sample (pd.Series): Sample data for diversity calculation.
        dropout (float, optional): Dropout rate for counterfactual generation. \
            These are the features that are set mutable. Defaults to 0.5.
        num_steps (int, optional): Number of optimization steps. Defaults to 20.
        batch_size (int, optional): Batch size for diversity calculation. \
            Defaults to 30.

    Returns:
        Dict[str, float]: Diversity metrics.
    """
    target_name = cf_optimizer.target_name
    sample = sample.to_frame().T
    counterfactual_template = generate_counterfactual_templates(
        sample, target_name, cf_optimizer.metadata, dropout=dropout
    )
    repeated_sample = pd.concat([sample] * batch_size, axis=0).reset_index(drop=True)
    repeated_cf_template = pd.concat(
        [counterfactual_template] * batch_size, axis=0
    ).reset_index(drop=True)

    counterfactuals = cf_optimizer.optimize_counterfactuals(
        repeated_sample,
        repeated_cf_template,
        target_name,
        num_steps,
        evaluate_each=0,
        verbose=False,
    )

    diversity = calculate_diversity(counterfactuals, metadata=cf_optimizer.metadata)

    return {
        "categorical_diversity": diversity["hamming_mean"].mean(),
        "continuous_diversity": diversity["shifts_mean"].mean(),
    }


def train_step(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Executes a single training step.

    Args:
        model (nn.Module): PyTorch model.
        dataloader (DataLoader): DataLoader for the training data.
        optimizer (optim.Optimizer): Optimizer for the model.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run the training on.

    Returns:
        float: Average training loss.
    """
    model.train()
    running_loss = 0.0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    average_loss = running_loss / len(dataloader)
    return average_loss


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluates the model on a given dataset.

    Args:
        model (nn.Module): PyTorch model.
        dataloader (DataLoader): DataLoader for the evaluation data.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run the evaluation on.

    Returns:
        Tuple[float, float]: Average evaluation loss and accuracy.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    average_loss = running_loss / len(dataloader)
    accuracy = correct / total

    return average_loss, accuracy


def proceed_and_write_config(
    config: Dict[str, Any], base_save_path: Union[str, Path], resume: bool
):
    """
    Checks configuration consistency and writes the config file if necessary.

    Args:
        config (Dict[str, Any]): Configuration dictionary.
        base_save_path (Union[str, Path]): Base path to save the configuration.
        resume (bool): Whether to resume training from the last checkpoint.

    Returns:
        bool: True if the configuration is consistent, False otherwise.
    """
    base_save_path = Path(base_save_path)
    config_path = base_save_path / "config.json"
    if resume and config_path.exists():
        if not same_config(load_json_file(config_path), config, ignore=["epochs"]):
            raise ValueError("Hyperparameters in config need to be the same to resume.")
        else:
            return True
    else:
        # handle existing files in save path
        if not proceed_if_filedir_exists(base_save_path):
            return False
        else:
            base_save_path.mkdir(parents=True, exist_ok=True)
            write_json_file(config_path, config)
            return True
