"""
This module provides utilities for loading, preprocessing, and transforming datasets 
for machine learning applications. It includes functionalities for handling 
Pandas DataFrames and converting them into PyTorch datasets, preparing data 
loaders, and downloading and preprocessing specific datasets.
"""

import json
import logging
import random
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union

import kaggle
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from ..utils.utils import remove_folder
from .data_preprocess import drop_na, infer_continuous_columns
from .transformer import CtganTransformer

logger = logging.getLogger(__name__)


class PandasDataset(Dataset):
    """
    A PyTorch Dataset for handling Pandas DataFrames.

    Args:
        dataframe (pd.DataFrame): The input data as a Pandas DataFrame.
        target_name (str): The name of the target column in the dataframe.
        transform (Callable[[pd.DataFrame], torch.Tensor]): The transformation function
            to apply to the dataframe.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        target_name: str,
        transform: Callable[[pd.DataFrame], torch.Tensor],
    ):
        self.features = transform(dataframe.drop(columns=[target_name])).to(
            torch.float32
        )
        self.targets = transform(dataframe[[target_name]]).argmax(dim=1)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]


class OnlyFeaturesDataset(Dataset):
    """
    A PyTorch Dataset for handling features only.

    Args:
        features (torch.Tensor): The input features as a PyTorch Tensor.
    """

    def __init__(self, features: torch.Tensor) -> None:
        super().__init__()
        self.features = features

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.features[idx]


def prepare_dataloaders(
    train: pd.DataFrame,
    test: pd.DataFrame,
    data_transformer: CtganTransformer,
    separate_target_name: Optional[str] = None,
    batch_size: int = 300,
) -> Tuple[DataLoader, DataLoader, CtganTransformer]:
    """
    Prepare DataLoaders for training and testing datasets.

    Args:
        train (pd.DataFrame): The training dataset.
        test (pd.DataFrame): The testing dataset.
        data_transformer (CtganTransformer): The data transformer object.
        separate_target_name (Optional[str]): The name of the target column, if any.
        batch_size (int): The batch size for the DataLoader.

    Returns:
        Tuple[DataLoader, DataLoader, CtganTransformer]: The training DataLoader,
            the testing DataLoader, and the data transformer.
    """
    features = train.columns.to_list()
    continuous_cols = infer_continuous_columns(train)
    categorical_cols = list(set(features) - set(continuous_cols))

    if not data_transformer.metadata:
        data_transformer.fit(raw_data=train, categorical_columns=categorical_cols)

    if separate_target_name:
        train_dataset = PandasDataset(
            train, separate_target_name, data_transformer.transform
        )
        test_dataset = PandasDataset(
            test, separate_target_name, data_transformer.transform
        )
    else:
        train_transformed = data_transformer.transform(train)
        test_transformed = data_transformer.transform(test)

        train_dataset = OnlyFeaturesDataset(train_transformed)
        test_dataset = OnlyFeaturesDataset(test_transformed)

    train_dataloader = DataLoader(
        train_dataset, batch_size, shuffle=True, drop_last=True
    )
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True, drop_last=True)

    return train_dataloader, test_dataloader, data_transformer


# Construct the path to the JSON file
JSON_PATH = Path(__file__).parent / "datasets.json"


def download_dataset(data_dir: Union[str, Path], dataset_name: str) -> pd.DataFrame:
    """
    Download a dataset from a predefined JSON configuration file.

    Args:
        dataset_name (str): The name of the dataset to download.

    Returns:
        pd.DataFrame: A Pandas DataFrames of the dataset.
    """
    with open(JSON_PATH, "r") as file:
        datasets = json.load(file)
    dataset = datasets[dataset_name]

    if dataset["source_type"] == "kaggle":
        kaggle.api.authenticate()

        # Download the dataset
        dir_path = Path(data_dir) / dataset_name
        kaggle.api.dataset_download_files(
            dataset["source"], path=dir_path, unzip=True, force=True
        )
        merged = []
        for file_name in dataset["merge_files"]:
            merged.append(pd.read_csv(dir_path / file_name, **dataset["read_csv_args"]))

        # Delete everything from the dir_path
        remove_folder(dir_path)
        merged = pd.concat(merged, axis=0)
        dir_path.mkdir(parents=True, exist_ok=True)
        preprocess_function = globals().get(f"preprocess_{dataset_name}")
        if preprocess_function:
            merged = preprocess_function(
                merged, dataset["column_names"], dataset["category_mappings"]
            )
        else:
            raise ValueError(
                f"No preprocess function found for dataset: {dataset_name}"
            )
        merged.to_csv(dir_path / dataset["file_name"], index=False)
    else:
        raise ValueError("Only kaggle datasets are supported.")

    return merged


def preprocess_adult(
    adult_data: pd.DataFrame,
    column_names: Dict[str, str],
    category_map: None,
    exclude_na: bool = True,
) -> pd.DataFrame:
    """
    Preprocess the Adult dataset.

    Args:
        adult_data (pd.DataFrame): The input Adult dataset.
        exclude_na (bool): Whether to exclude rows with missing values.

    Returns:
        pd.DataFrame: The preprocessed Adult dataset.
    """
    if exclude_na:
        adult_data = drop_na(adult_data)

    adult_data = adult_data.rename(columns=column_names)
    adult_data = adult_data.drop(columns=["fnlwgt"])

    return adult_data


def preprocess_heart_disease(
    heart_data: pd.DataFrame,
    column_names: Dict[str, str],
    category_map: None,
    exclude_na: bool = True,
    target_name: str = "Heart_Disease",
) -> pd.DataFrame:
    """
    Preprocess the Heart Disease dataset.

    Args:
        heart_data (pd.DataFrame): The input Heart Disease dataset.
        exclude_na (bool): Whether to exclude rows with missing values.
        target_name (str): The name of the target column.

    Returns:
        pd.DataFrame: The preprocessed Heart Disease dataset.
    """
    if exclude_na:
        heart_data = drop_na(heart_data)

    heart_data = heart_data.rename(columns=column_names)

    # reorder to ensure target is the last column
    features = list(heart_data.columns)
    features.remove(target_name)
    new_cols = features + [target_name]

    return heart_data[new_cols]


def preprocess_diabetes(
    diabetes_data: pd.DataFrame,
    column_names: Dict[str, str],
    category_map: Dict[str, Dict[str, str]],
    exclude_na: bool = True,
    target_name: str = "Diabetes",
) -> pd.DataFrame:
    if exclude_na:
        diabetes_data = drop_na(diabetes_data)

    diabetes_data = diabetes_data.rename(columns=column_names)

    # Apply a category map to the categorical columns
    for column, mapping in category_map.items():
        diabetes_data[column] = diabetes_data[column].map(
            lambda x: mapping[str(int(x))] if pd.notnull(x) else x
        )

    # reorder to ensure target is the last column
    features = list(diabetes_data.columns)
    features.remove(target_name)
    new_cols = features + [target_name]

    return diabetes_data[new_cols]


def preprocess_employees(
    employee_data: pd.DataFrame,
    column_names: Dict[str, str],
    category_map: None,
    exclude_na: bool = True,
) -> pd.DataFrame:
    if exclude_na:
        employee_data = drop_na(employee_data)

    employee_data = employee_data.rename(columns=column_names)
    employee_data = employee_data.drop(columns=["Employee_ID"])

    return employee_data


def switch_adult_target(target: str) -> str:
    """
    Switch the target value for the Adult dataset.

    Args:
        target (str): The current target value.

    Returns:
        str: The switched target value.
    """
    if target == ">50K":
        return "<=50K"
    else:
        return ">50K"


def switch_heart_disease_target(target: str) -> str:
    """
    Switch the target value for the Heart Disease dataset.

    Args:
        target (str): The current target value.

    Returns:
        str: The switched target value.
    """
    if target == "Yes":
        return "No"
    else:
        return "Yes"


def switch_diabetes_target(target: str) -> str:
    """
    Switch the target value for the Diabetes dataset.

    Args:
        target (str): The current target value.

    Returns:
        str: The switched target value.
    """
    targets = ["No diabetes", "Prediabetes", "Diabetes"]
    return random.choice([t for t in targets if t != target])


def switch_employees_target(target: str) -> str:
    """
    Switch the target value for the Employees dataset.

    Args:
        target (str): The current target value.

    Returns:
        str: The switched target value.
    """
    if target == "Stayed":
        return "Left"
    else:
        return "Stayed"
