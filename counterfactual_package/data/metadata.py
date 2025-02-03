"""
MetaData module: to easily access metadata of a transformed mixed dataset.
"""

from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Literal, Optional, Sequence, Union

import torch

DataTypeElement = Literal[
    "continuous", "categorical", "one_hot", "float", "int", "string"
]


class DataTypes:
    """Class containing constants for different data types."""

    CONTINUOUS: DataTypeElement = "continuous"
    CATEGORICAL: DataTypeElement = "categorical"
    ONEHOT: DataTypeElement = "one_hot"
    FLOAT: DataTypeElement = "float"
    INT: DataTypeElement = "int"
    STRING: DataTypeElement = "string"


@dataclass
class DataRepresentation:
    """Class representing the data type and dimension of a dataset column."""

    dimension: int
    datatype: DataTypeElement


@dataclass
class ColumnMeta:
    """
    Class representing metadata of a dataset column.

    Attributes:
        column_name (str): Name of the column.
        column_type (DataTypeElement): Type of the column.
        transform_function (Any): Transformation function applied to the column.
        output_dimension (int): Dimension of the transformed column.
        datarepresentations (List[DataRepresentation]): List of data representations.
        original_dtype (Optional[Any]): Original data type of the column.
        value_frequencies (Optional[Dict[str, float]]): Value frequencies in the column.
        percentile_values (Optional[List[float]]): Percentile values of the column.
        mean (Optional[float]): Mean value of the column.
        std (Optional[float]): Standard deviation of the column.
    """

    column_name: str
    column_type: DataTypeElement
    transform_function: Any
    output_dimension: int
    datarepresentations: List[DataRepresentation]
    original_dtype: Optional[Any] = None
    value_frequencies: Optional[Dict[str, float]] = None
    percentile_values: Optional[List[float]] = None
    mean: Optional[float] = None
    std: Optional[float] = None


class MetaData(DataTypes):
    """
    Class for handling metadata of a transformed mixed dataset.

    Attributes:
        dataset_name (Optional[str]): Name of the dataset.
        _meta_list (List[ColumnMeta]): List of column metadata.
        _column_to_idx (Dict[str, int]): Dictionary mapping column names to indices.
        _column_to_cat_idx (Dict[str, int]): Dictionary mapping categorical column
                                             names to indices.
    """

    def __init__(
        self, meta_list: List[ColumnMeta], dataset_name: Optional[str] = None
    ) -> None:
        """
        Initialize the MetaData object.

        Args:
            meta_list (List[ColumnMeta]): List of column metadata.
            dataset_name (Optional[str]): Name of the dataset.
        """
        self.dataset_name = dataset_name
        self._meta_list = meta_list
        self._column_to_idx = self._initialize_column_idx_dict(meta_list)
        self._column_to_cat_idx = self._initialize_column_idx_dict(
            meta_list, only_categories=True
        )

    def num_columns(self) -> int:
        """Return the number of columns in the metadata."""
        return len(self)

    def num_transformed_columns(self) -> int:
        """Return the total number of transformed columns."""
        return sum([column.output_dimension for column in self._meta_list])

    def conditional_column_vector(
        self, columns: Sequence[Union[str, int]] = ()
    ) -> torch.Tensor:
        """
        Create a boolean vector indicating which columns are conditional.

        Args:
            columns (Sequence[Union[str, int]]): Sequence of column names or indices.

        Returns:
            torch.Tensor: Boolean tensor indicating conditional columns.
        """
        conditional = torch.zeros([len(self)], dtype=torch.bool)

        if all(isinstance(e, int) for e in columns):
            conditional[list(columns)] = True
        elif all(isinstance(e, str) for e in columns):
            for column_name in columns:
                index = self.column_to_idx(str(column_name))
                conditional[index] = True
        else:
            raise ValueError(
                "columns variable should contain indices (int) or column names (str)"
            )
        return conditional

    def column_to_idx(self, column: str, only_categories: bool = False) -> int:
        """
        Get the index of a column by name.

        Args:
            column (str): Column name.
            only_categories (bool): If True, only search categorical columns.

        Returns:
            int: Index of the column.
        """
        if only_categories:
            return self._column_to_cat_idx[column]
        else:
            return self._column_to_idx[column]

    def has_column(self, column_name: str) -> bool:
        """Check if a column exists in the metadata."""
        return column_name in self._column_to_idx

    def column_to_transformed_idxs(self, column: str) -> List[int]:
        """
        Get the indices of the transformed columns for a given column.

        Args:
            column (str): Column name.

        Returns:
            List[int]: List of indices for the transformed columns.
        """
        start = 0
        for search_col in self:
            if search_col.column_name == column:
                end = start + search_col.output_dimension
                return [i for i in range(start, end)]
            else:
                start += search_col.output_dimension
        else:
            raise ValueError(f"{column} not in the dataset.")

    def _initialize_column_idx_dict(
        self, meta_list: List[ColumnMeta], only_categories: bool = False
    ) -> Dict[str, int]:
        """
        Initialize a dictionary mapping column names to indices.

        Args:
            meta_list (List[ColumnMeta]): List of column metadata.
            only_categories (bool): If True, only include categorical columns.

        Returns:
            Dict[str, int]: Dictionary mapping column names to indices.
        """
        if only_categories:
            categories = [
                column.column_name
                for i, column in enumerate(meta_list)
                if column.column_type == DataTypes.CATEGORICAL
            ]
            return {cat_name: i for i, cat_name in enumerate(categories)}
        else:
            return {column.column_name: i for i, column in enumerate(meta_list)}

    def __getitem__(self, item: Union[int, str]) -> ColumnMeta:
        """
        Get column metadata by index or name.

        Args:
            item (Union[int, str]): Column index or name.

        Returns:
            ColumnMeta: Metadata of the specified column.
        """
        if isinstance(item, int):
            return self._meta_list[item]
        else:
            return self._meta_list[self.column_to_idx(item)]

    def __len__(self) -> int:
        """Return the number of columns in the metadata."""
        return len(self._meta_list)

    def __iter__(self) -> Generator[ColumnMeta, None, None]:
        """Iterate over the column metadata."""
        for column in self._meta_list:
            yield column

    def __str__(self) -> str:
        """Return a string representation of the metadata."""
        description = f"Metadata {self.dataset_name} of {len(self)} columns.\n\n"

        for i, column in enumerate(self._meta_list):
            description += f"{i}.  {column.column_name} ({column.column_type} - \
                {column.original_dtype}): dim={column.output_dimension}\n"
            for repr in column.datarepresentations:
                description += f"         {repr.datatype}: dim={repr.dimension}\n"

        return description

    def __repr__(self) -> str:
        """Return a detailed string representation of the metadata."""
        description = "{"
        for column in self._meta_list:
            description += f"{column.__repr__()}\n"
        description += f"name={self.dataset_name}, columns={len(self)}"
        description += "}"

        return description
