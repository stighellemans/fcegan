"""
This module provides functions for splitting and validating datasets, inferring column 
types, handling missing values, and converting data formats. It supports pandas 
DataFrame, pandas Series, numpy ndarray, and torch Tensor.
"""

from typing import Dict, List, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np
import pandas as pd
import torch

Data = TypeVar("Data", pd.DataFrame, pd.Series, np.ndarray, torch.Tensor)


def split_by_indices(
    matrix: Data,
    split_indices: List[np.ndarray],
) -> List[Data]:
    """
    Split data into multiple parts based on given indices.

    Args:
        matrix (Data): The data to be split, which can be a DataFrame, Series,
                       numpy array, or torch tensor.
        split_indices (List[np.ndarray]): List of numpy arrays containing indices
                                          to split the data.

    Returns:
        List[Data]: List of data parts split according to the indices.
    """
    if isinstance(matrix, (pd.DataFrame, pd.Series)):
        return [matrix.iloc[idx] for idx in split_indices]
    elif isinstance(matrix, np.ndarray):
        return [matrix[idx] for idx in split_indices]
    elif isinstance(matrix, torch.Tensor):
        return [matrix[torch.from_numpy(idx)] for idx in split_indices]
    else:
        raise TypeError("Unsupported data type")


def valid_data_split(
    data: pd.DataFrame,
    categorical_cols: Sequence[str],
    val_prop: float = 0.2,
    test_prop: float = 0.2,
    shuffle: bool = True,
    random_state: Optional[int] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Split data into training, validation, and test sets ensuring valid training data.

    Args:
        data (pd.DataFrame): The data to be split.
        categorical_cols (Sequence[str]): List of categorical column names.
        val_prop (float, optional): Proportion of validation data. Default is 0.2.
        test_prop (float, optional): Proportion of test data. Default is 0.2.
        shuffle (bool, optional): Whether to shuffle the data before splitting.
                                  Default is True.
        random_state (Optional[int], optional): Random seed for reproducibility.
                                                Default is None.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary with 'train', 'val', and 'test' DataFrames.
    """
    if not random_state:
        random_state = np.random.randint(0, 2**32)

    train_prop = 1 - val_prop - test_prop
    seeking_split = True

    while seeking_split:
        splitted_data = split_data(
            data=data,
            proportions=[train_prop, val_prop, test_prop],
            shuffle=shuffle,
            random_state=random_state,
        )

        train, val, test = splitted_data

        if valid_train(
            train=train,
            others=[val, test],
            categorical_columns=categorical_cols,
        ):
            seeking_split = False
        else:
            print(
                f"random state {random_state} gives no valid split, "
                f"trying random state {random_state+1}"
            )
            random_state += 1

    return {"train": train, "val": val, "test": test}


def split_data(
    data: Data,
    proportions: Sequence[float],
    shuffle: bool = True,
    random_state: Optional[int] = None,
) -> List[Data]:
    """
    Split data into multiple parts based on given proportions.

    Args:
        data (Data): The data to be split, which can be a DataFrame, Series,
                     numpy array, or torch tensor.
        proportions (Sequence[float]): Proportions for splitting the data.
        shuffle (bool, optional): Whether to shuffle the data before splitting.
                                  Default is True.
        random_state (Optional[int], optional): Random seed for reproducibility.
                                                Default is None.

    Returns:
        List[Data]: List of data parts split according to the proportions.
    """
    if any([proportion < 0 or proportion > 1 for proportion in proportions]):
        raise ValueError("Proportions should be between [0,1].")

    if sum(proportions) != 1:
        raise ValueError("Sum of proportions should equal to one.")

    if not random_state:
        random_state = np.random.randint(0, 2**32)

    # Create an array of indices and shuffle it
    indices = np.arange(len(data))
    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)

    # Calculate the split points based on the proportions
    # Exclude the last proportion
    cumsum = np.cumsum(proportions[:-1])
    split_points = (cumsum * len(data)).astype(int)

    data_splits = []
    # Split the shuffled indices and then use them to create the data/target splits
    split_indices = np.split(indices, split_points)
    data_splits = split_by_indices(data, split_indices)

    return data_splits


def valid_train(
    train: pd.DataFrame,
    others: Sequence[pd.DataFrame],
    categorical_columns: Sequence[str],
) -> bool:
    """
    Validate that training data contains all categories present in other data splits.

    Args:
        train (pd.DataFrame): Training data.
        others (Sequence[pd.DataFrame]): Other data splits (validation and test).
        categorical_columns (Sequence[str]): List of categorical column names.

    Returns:
        bool: True if training data contains all categories from other splits, \
            False otherwise.
    """
    other_df = pd.concat(others, axis=0)

    for column in categorical_columns:
        train_unique_values = set(train[column].unique())
        other_unique_values = set(other_df[column].unique())

        if not other_unique_values.issubset(train_unique_values):
            return False

    return True


def infer_continuous_columns(data: pd.DataFrame) -> List[str]:
    """
    Infer continuous (numerical) columns from a DataFrame.

    Args:
        data (pd.DataFrame): The input data.

    Returns:
        List[str]: List of continuous column names.
    """
    numerical_data = data.select_dtypes(include="number")

    if numerical_data.empty:
        return []
    else:
        return numerical_data.columns.to_list()


def infer_column_types(dataframe: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Infer column types (categorical and continuous) from a DataFrame.

    Args:
        dataframe (pd.DataFrame): The input data.

    Returns:
        Tuple[List[str], List[str]]: Tuple containing list of categorical column names
                                     and list of continuous column names.
    """
    continuous_cols = infer_continuous_columns(dataframe)
    categorical_cols = list(set(dataframe.columns) - set(continuous_cols))

    return categorical_cols, continuous_cols


def na_present(data: Data) -> bool:
    """
    Check if there are any missing values in the data.

    Args:
        data (Data): The data to check, which can be a DataFrame, Series,
                     numpy array, or torch tensor.

    Returns:
        bool: True if there are missing values, False otherwise.
    """
    if isinstance(data, pd.Series):
        nan_included = data.isna().any()
    elif isinstance(data, pd.DataFrame):
        nan_included = data.isna().any().any()
    elif isinstance(data, np.ndarray):
        nan_included = np.isnan(data).any()
    elif isinstance(data, torch.Tensor):
        nan_included = bool(torch.isnan(data).any())
    else:
        raise ValueError(f"Datatype {type(data)} not supported")

    return nan_included


PandasData = TypeVar("PandasData", pd.DataFrame, pd.Series)


def drop_na(data: PandasData) -> PandasData:
    """
    Drop rows with missing values and reset the index.

    Args:
        data (PandasData): The data to clean, which can be a DataFrame or Series.

    Returns:
        PandasData: The cleaned data with missing values dropped.
    """
    data = data.dropna()
    data = data.reset_index(drop=True)
    return data


def one_hot_encode_column(
    data_column: pd.Series, prefix_sep: str = "+"
) -> pd.DataFrame:
    """
    Perform one-hot encoding on a column.

    Args:
        data_column (pd.Series): The column to one-hot encode.
        prefix_sep (str, optional): Separator to use for the prefix. Default is "+".

    Returns:
        pd.DataFrame: One-hot encoded DataFrame.
    """
    return pd.get_dummies(
        data_column,
        prefix=str(data_column.name),
        prefix_sep=prefix_sep,
    )


def to_tensor(data: Union[pd.DataFrame, pd.Series, np.ndarray]) -> torch.Tensor:
    """
    Convert data to a torch tensor.

    Args:
        data (Union[pd.DataFrame, pd.Series, np.ndarray]): The data to convert.

    Returns:
        torch.Tensor: The converted torch tensor.
    """
    if isinstance(data, (pd.DataFrame, pd.Series)):
        return torch.Tensor(data.values.astype(float))
    elif isinstance(data, np.ndarray):
        return torch.Tensor(data.astype(float))
    else:
        raise ValueError(f"Datatype {type(data)} not supported")
