"""
This module contains functions for manipulating and analyzing data tensors,
dataframes, and metadata in the context of machine learning. It includes
operations such as splitting tensors, transforming masks, applying masks, 
corrupting dataframes, generating counterfactuals, and calculating distances 
and diversity metrics.
"""

import random
from functools import partial
from typing import Dict, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch

from .metadata import DataTypes, MetaData


def split_torch_tensor_by_columns(
    tensor: torch.Tensor, selected_indices: Sequence[int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split a torch tensor into two tensors based on selected column indices.

    Args:
        tensor (torch.Tensor): The input tensor.
        selected_indices (Sequence[int]): Indices of columns to select.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Selected columns tensor and
        remaining columns tensor.
    """
    selected_columns = tensor[:, sorted(selected_indices)]

    # Selecting the rest of the columns
    all_indices = set(range(tensor.shape[1]))
    rest_indices = list(all_indices - set(selected_indices))
    rest_indices.sort()
    rest_columns = tensor[:, rest_indices]

    return selected_columns, rest_columns


def transform_mask(mask: torch.Tensor, metadata: MetaData) -> torch.Tensor:
    """
    Transform a mask tensor based on metadata.

    Args:
        mask (torch.Tensor): The input mask tensor.
        metadata (MetaData): Metadata describing the transformation.

    Returns:
        torch.Tensor: Transformed mask tensor.
    """
    mask_transformed = torch.zeros(
        (mask.shape[0], metadata.num_transformed_columns()), dtype=torch.bool
    )

    start = 0
    for i, column in enumerate(metadata):
        end = start + column.output_dimension
        mask_transformed[:, start:end] = mask[:, i].unsqueeze(1)
        start = end

    return mask_transformed


def apply_mask(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Apply a mask to a tensor, setting masked elements to zero.

    Args:
        tensor (torch.Tensor): The input tensor.
        mask (torch.Tensor): The mask tensor.

    Returns:
        torch.Tensor: The masked tensor.
    """
    tensor = tensor.clone()
    tensor[~mask] = 0.0

    return tensor


def choose_other_category(value: str, column_name: str, metadata: MetaData) -> str:
    """
    Choose a random category different from the given value.

    Args:
        value (str): The current value.
        column_name (str): The column name in the metadata.
        metadata (MetaData): Metadata containing value frequencies.

    Returns:
        str: A randomly chosen different category.
    """
    values = list(metadata[column_name].value_frequencies)
    values.remove(value)
    return random.choice(values)


def randomly_change_to_other_cat(
    data: torch.Tensor, column: str, metadata: MetaData
) -> torch.Tensor:
    """
    Randomly change elements of a column to other category values.

    Args:
        data (torch.Tensor): The input data tensor.
        column (str): The column name.
        metadata (MetaData): Metadata describing the column.

    Returns:
        torch.Tensor: The modified tensor.
    """
    idxs = metadata.column_to_transformed_idxs(column)

    new_tensor = data.clone()
    to_change = new_tensor[:, idxs]
    n_rows = to_change.shape[0]

    new_selections = randomly_get_other_value_ids(data, column, metadata)

    # Update the new_tensor accordingly
    changed = torch.zeros_like(to_change)
    changed[torch.arange(n_rows), new_selections] = 1.0
    new_tensor[:, idxs] = changed

    return new_tensor


def randomly_get_other_value_ids(
    data: torch.Tensor, column: str, metadata: MetaData
) -> torch.Tensor:
    """
    Get indices of random other values for a column in the tensor.

    Args:
        data (torch.Tensor): The input data tensor.
        column (str): The column name.
        metadata (MetaData): Metadata describing the column.

    Returns:
        torch.Tensor: Indices of other values.
    """
    # Get the indices for the specified column
    idxs = metadata.column_to_transformed_idxs(column)
    column_tensor = data[:, idxs]
    n_rows, n_cols = column_tensor.shape

    # Find current '1' positions
    current_selections = column_tensor.argmax(dim=1)

    # Create a tensor of all possible indices
    all_indices = torch.arange(n_cols).repeat(n_rows, 1)

    # Mask out the current selections
    mask = torch.ones_like(all_indices, dtype=torch.bool)
    mask[torch.arange(n_rows), current_selections] = False
    filtered_indices = all_indices[mask].view(n_rows, n_cols - 1)

    # Generate random new positions for each row
    random_indices = torch.randint(0, n_cols - 1, (n_rows,))
    new_selections = filtered_indices[torch.arange(n_rows), random_indices]

    return new_selections


def corrupt_dataframe_with_nan(
    df: Union[pd.DataFrame, pd.Series],
    corruption_percentage: float,
    exclude: Sequence[str] = (),
) -> pd.DataFrame:
    """
    Corrupt a dataframe with NaN values randomly.

    Args:
        df (Union[pd.DataFrame, pd.Series]): The input dataframe or series.
        corruption_percentage (float): The percentage of columns to corrupt.
        exclude (Sequence[str], optional): Columns to exclude from corruption.

    Returns:
        pd.DataFrame: The corrupted dataframe.
    """
    df_corrupt = df.copy()

    if isinstance(df_corrupt, pd.Series):
        df_corrupt = df_corrupt.to_frame().T

    include_columns = [col for col in df.columns if col not in exclude]

    if not include_columns:
        return df_corrupt

    num_columns_to_corrupt = int(len(include_columns) * corruption_percentage)

    for idx in df_corrupt.index:
        columns_to_corrupt = np.random.choice(
            include_columns, num_columns_to_corrupt, replace=False
        )
        for col in columns_to_corrupt:
            df_corrupt.at[idx, col] = np.nan

    return df_corrupt


def dynamically_corrupt_dataframe_with_nan(
    df: Union[pd.DataFrame, pd.Series],
    mean_corruption_percentage: float,
    exclude: Sequence[str] = (),
) -> pd.DataFrame:
    """
    Dynamically corrupt a dataframe with NaN values.

    Args:
        df (Union[pd.DataFrame, pd.Series]): The input dataframe or series.
        mean_corruption_percentage (float): The mean percentage of values to corrupt.
        exclude (Sequence[str], optional): Columns to exclude from corruption.

    Returns:
        pd.DataFrame: The corrupted dataframe.
    """
    df_corrupt = df.copy()

    if isinstance(df_corrupt, pd.Series):
        df_corrupt = df_corrupt.to_frame().T

    include_columns = [col for col in df.columns if col not in exclude]

    if not include_columns:
        return df_corrupt

    indices = [
        (i, df.columns.get_loc(col))
        for i in range(df.shape[0])
        for col in include_columns
    ]
    num_values_to_corrupt = int(len(indices) * mean_corruption_percentage)
    indices_to_corrupt = np.random.choice(
        len(indices), num_values_to_corrupt, replace=False
    )

    for idx in indices_to_corrupt:
        row, col = indices[idx]
        df_corrupt.iat[row, col] = np.nan

    return df_corrupt


def generate_counterfactual_masks(
    dropout: Union[float, Tuple[float, float]],
    batch_size: int,
    metadata: MetaData,
    no_dropout_cols: Sequence[str] = (),
) -> torch.Tensor:
    """
    Generate masks for counterfactual generation with specified dropout rates.

    Args:
        dropout (Union[float, Tuple[float, float]]): Dropout rate or range.
        batch_size (int): Batch size for mask generation.
        metadata (MetaData): Metadata describing the columns.
        no_dropout_cols (Sequence[str], optional): Columns to exclude from dropout.

    Returns:
        torch.Tensor: Generated masks.
    """
    # Create dropout tensor based on the type of dropout input
    if isinstance(dropout, Tuple):
        dropout_tensor = torch.empty(batch_size).uniform_(dropout[0], dropout[1])
    else:
        dropout_tensor = torch.full((batch_size,), dropout)

    n_cols = metadata.num_columns()
    n_drops = (n_cols * dropout_tensor).round().long()  # Number of columns to drop
    n_keep = n_cols - n_drops  # Number of columns to keep

    # Create a random tensor for the entire batch
    rand_tensor = torch.rand((batch_size, n_cols))

    # Generate a mask with the required number of columns to keep
    mask = torch.zeros((batch_size, n_cols), dtype=torch.bool)

    # Sort the random tensor to get the indices of columns to keep
    sorted_indices = torch.argsort(rand_tensor, dim=1, descending=True)
    keep_indices = sorted_indices[:, : n_keep.max().item()]
    batch_indices = torch.arange(batch_size).unsqueeze(1).expand_as(keep_indices)
    mask[batch_indices, keep_indices] = True

    # Ensure specified columns are always kept
    if no_dropout_cols:
        for col in no_dropout_cols:
            idx = metadata.column_to_idx(col)
            mask[:, idx] = True

    return mask


def generate_counterfactual_templates(
    originals: pd.DataFrame, target_name: str, metadata: MetaData, dropout: float = 0.5
) -> pd.DataFrame:
    """
    Generate counterfactual templates by corrupting the originals.

    Args:
        originals (pd.DataFrame): Original data.
        target_name (str): Target column name.
        metadata (MetaData): Metadata describing the columns.
        dropout (float, optional): Dropout rate for corruption. Defaults to 0.5.

    Returns:
        pd.DataFrame: Counterfactual templates.
    """
    templates = dynamically_corrupt_dataframe_with_nan(
        originals, dropout, exclude=[target_name]
    )
    target_changer = partial(
        choose_other_category, column_name=target_name, metadata=metadata
    )
    templates[target_name] = templates[target_name].apply(target_changer)

    return templates


def cf_template_to_keep_mask(
    cf_templates: torch.Tensor, metadata: MetaData, transformed: bool = False
) -> torch.Tensor:
    """
    Convert counterfactual templates to a keep mask.

    Args:
        cf_templates (torch.Tensor): Counterfactual templates.
        metadata (MetaData): Metadata describing the columns.
        transformed (bool, optional): Whether to transform the mask.
        Defaults to False.

    Returns:
        torch.Tensor: Keep mask.
    """
    to_keep_list = []
    for column_meta in metadata:
        column_name = column_meta.column_name
        idxs = metadata.column_to_transformed_idxs(column_name)

        to_keep_list.append(cf_templates[:, idxs].sum(dim=1, keepdim=True) != 0)
    to_keep = torch.concat(to_keep_list, dim=1)

    if transformed:
        return transform_mask(to_keep, metadata)
    else:
        return to_keep


def calculate_hamming_distance(
    df1: Union[pd.DataFrame, pd.Series],
    df2: Union[pd.DataFrame, pd.Series],
    stds: Dict[str, float],
    threshold: float = 0.01,
) -> pd.DataFrame:
    """
    Calculate the Hamming distance between two dataframes or series.

    Args:
        df1 (Union[pd.DataFrame, pd.Series]): The first dataframe or series.
        df2 (Union[pd.DataFrame, pd.Series]): The second dataframe or series.
        stds (Dict[str, float]): Standard deviations for continuous columns.
        threshold (float, optional): Threshold for continuous columns.
        Defaults to 0.01.

    Returns:
        pd.DataFrame: DataFrame with Hamming distances and comparisons.
    """
    # TO BE OMITTED

    if isinstance(df1, pd.Series):
        df1 = df1.to_frame().T

    if isinstance(df2, pd.Series):
        df2 = df2.to_frame().T

    cat_hamming_distances = []
    cat_n_comparisons = []
    cont_hamming_distances = []
    cont_n_comparisons = []

    n_rows = df1.shape[0]

    for i in range(n_rows):
        row1 = df1.iloc[i]
        row2 = df2.iloc[i]

        cat_hamming_distances.append(0)
        cat_n_comparisons.append(0)
        cont_hamming_distances.append(0)
        cont_n_comparisons.append(0)

        for col in df1.columns:
            val1 = row1[col]
            val2 = row2[col]

            # Skip if val1 is NaN
            if pd.isna(val1) or pd.isna(val2):
                continue
            elif col in stds:
                # Continuous columns
                cont_n_comparisons[-1] += 1
                if abs(val1 - val2) / stds[col] > threshold:
                    cont_hamming_distances[-1] += 1
            else:
                # Categorical columns
                cat_n_comparisons[-1] += 1
                if val1 != val2:
                    cat_hamming_distances[-1] += 1

    hamming = pd.DataFrame(
        {
            "cat_hamming": cat_hamming_distances,
            "cat_comparisons": cat_n_comparisons,
            "cont_hamming": cont_hamming_distances,
            "cont_comparisons": cont_n_comparisons,
        }
    )
    hamming["hamming"] = hamming["cat_hamming"] + hamming["cont_hamming"]
    hamming["comparisons"] = hamming["cat_comparisons"] + hamming["cont_comparisons"]

    return hamming


def calculate_identity(hamming_dist: int, total: int) -> float:
    """
    Calculate the identity score based on Hamming distance.

    Args:
        hamming_dist (int): The Hamming distance.
        total (int): The total number of comparisons.

    Returns:
        float: The identity score.
    """
    return (total - hamming_dist) / total


Number = Union[int, float]


def estimate_cdf(value: Number, percentile_values: Sequence[Number]) -> np.ndarray:
    """
    Estimate the CDF value for a given value and percentile values.

    Args:
        value (Number): The value to estimate CDF for.
        percentile_values (Sequence[Number]): The percentile values.

    Returns:
        np.ndarray: The estimated CDF value.
    """
    return np.interp(value, percentile_values, np.linspace(0, 1, 100))


def calculate_percentile_shift(
    start: Number, end: Number, percentile_values: Sequence[Number]
) -> np.ndarray:
    """
    Calculate the percentile shift between start and end values.

    Args:
        start (Number): The starting value.
        end (Number): The ending value.
        percentile_values (Sequence[Number]): The percentile values.

    Returns:
        np.ndarray: The percentile shift.
    """
    return estimate_cdf(end, percentile_values) - estimate_cdf(start, percentile_values)


def calculate_distances(
    start_df: Union[pd.DataFrame, pd.Series],
    end_df: Union[pd.DataFrame, pd.Series],
    metadata: MetaData,
) -> Dict[str, pd.DataFrame]:
    """
    Calculate the Hamming and percentile shift distances between two dataframes.

    Args:
        start_df (Union[pd.DataFrame, pd.Series]): The starting dataframe or series.
        end_df (Union[pd.DataFrame, pd.Series]): The ending dataframe or series.
        metadata (MetaData): Metadata describing the columns.

    Returns:
        Dict[str, pd.DataFrame]: Distances for each column.
    """
    hamming_matrix = {}
    percentile_shift_matrix = {}

    if isinstance(start_df, pd.Series):
        start_df = start_df.to_frame().T

    if isinstance(end_df, pd.Series):
        end_df = end_df.to_frame().T

    if set(start_df.columns) != set(end_df.columns):
        raise ValueError("Comparison between two different dataframes.")

    for col in start_df.columns:
        if not metadata.has_column(col):
            raise ValueError(
                f"Distance could not be calculated since '{col}' is not recognized."
            )
        col_meta = metadata[col]

        col1 = start_df[col].reset_index(drop=True)
        col2 = end_df[col].reset_index(drop=True)

        if col_meta.column_type == DataTypes.CONTINUOUS:
            percentile_shift_matrix[col] = col1.combine(
                col2,
                partial(
                    calculate_percentile_shift,
                    percentile_values=col_meta.percentile_values,
                ),
            )
        else:
            hamming_matrix[col] = (col1 != col2).astype(int)

    return {
        "hamming": pd.DataFrame(hamming_matrix),
        "percentile_shifts": pd.DataFrame(percentile_shift_matrix),
    }


def calculate_diversity(
    counterfactuals: pd.DataFrame, metadata: MetaData
) -> Dict[str, pd.DataFrame]:
    """
    Calculate the diversity of counterfactuals based on Hamming and shifts.

    Args:
        counterfactuals (pd.DataFrame): Dataframe of counterfactuals.
        metadata (MetaData): Metadata describing the columns.

    Returns:
        Dict[str, pd.DataFrame]: Mean Hamming and shifts metrics.
    """
    hamming_total = 0
    shifts_total = 0
    for i in range(1, len(counterfactuals)):
        comparisor = pd.concat([counterfactuals.iloc[i:], counterfactuals.iloc[:i]])
        hamming, shifts = calculate_distances(
            counterfactuals, comparisor, metadata
        ).values()
        hamming_total += hamming.mean(axis=0)
        shifts_total += shifts.abs().mean(axis=0)

    hamming_mean = hamming_total / (len(counterfactuals) - 1)
    shifts_mean = shifts_total / (len(counterfactuals) - 1)

    return {"hamming_mean": hamming_mean, "shifts_mean": shifts_mean}
