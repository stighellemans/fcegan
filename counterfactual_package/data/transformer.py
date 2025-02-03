from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed  # type: ignore
from rdt.transformers import ClusterBasedNormalizer, OneHotEncoder  # type: ignore

from .data_preprocess import infer_column_types
from .metadata import ColumnMeta, DataRepresentation, DataTypes, MetaData


class CtganTransformer:
    """Data Transformer.

    Possible to be used with NaN values.

    Model continuous columns with a BayesianGMM and normalize them to a scalar between
    [-1, 1] and a vector. Categorical columns are encoded using a OneHotEncoder.
    """

    output_dimension: Optional[int]
    _column_raw_dtypes: Optional[Dict[str, Any]] = None
    metadata: Optional[MetaData] = None
    dataframe = True

    def __init__(self, max_clusters: int = 10, weight_threshold: float = 0.005) -> None:
        """Create a data transformer.

        Args:
            max_clusters (int): Maximum number of Gaussian distributions in \
                Bayesian GMM.
            weight_threshold (float): Weight threshold for a Gaussian distribution \
                to be kept.
        """
        self._max_clusters = max_clusters
        self._weight_threshold = weight_threshold

    def fit(
        self,
        raw_data: pd.DataFrame,
        categorical_columns: Optional[Iterable[str]] = None,
    ) -> None:
        """Fit the ``DataTransformer``.

        Fits a ``ClusterBasedNormalizer`` for continuous columns and a
        ``OneHotEncoder`` for categorical columns.

        This step also counts the #columns in matrix data and meta data.

        Args:
            raw_data (pd.DataFrame): The raw input data.
            categorical_columns (Optional[Iterable[str]]): List of categorical \
                column names.
        """
        self.output_dimension = 0

        meta_list = []

        if not categorical_columns:
            categorical_columns, _ = infer_column_types(raw_data)

        self._column_raw_dtypes = {
            col: dtype
            for col, dtype in zip(raw_data.columns, raw_data.infer_objects().dtypes)
        }

        for column_name in raw_data.columns:
            if column_name in categorical_columns:
                column_meta = self._fit_categorical(raw_data[[column_name]])
            else:
                column_meta = self._fit_continuous(raw_data[[column_name]])

            meta_list.append(column_meta)

        self.metadata = MetaData(meta_list)

        for column_meta in self.metadata:
            self.output_dimension += column_meta.output_dimension

    def transform(self, raw_data: Union[pd.DataFrame, pd.Series]) -> torch.Tensor:
        """Take raw data and output a matrix data.

        Args:
            raw_data (Union[pd.DataFrame, pd.Series]): The raw input data.

        Returns:
            torch.Tensor: Transformed tensor data.
        """
        if isinstance(raw_data, pd.Series):
            raw_data = raw_data.to_frame()

        # Only use parallelization with larger data sizes.
        # Otherwise, the transformation will be slower.
        if raw_data.shape[0] < 500:
            column_data_list = self._synchronous_transform(raw_data, self.metadata)
        else:
            column_data_list = self._parallel_transform(raw_data, self.metadata)

        transformed = np.concatenate(column_data_list, axis=1).astype(float)

        return torch.tensor(transformed, dtype=torch.float32)

    def reverse_transform(
        self,
        data: Union[np.ndarray, torch.Tensor],
    ) -> pd.DataFrame:
        """Take matrix data and output raw data.

        Output uses the same type as input to the transform function.
        Either np array or pd dataframe.

        Args:
            data (Union[np.ndarray, torch.Tensor]): Transformed data.

        Returns:
            pd.DataFrame: Reversed raw data.
        """
        st = 0
        recovered_column_data_list = []
        column_names = []
        for column_meta in self.metadata:
            dim = column_meta.output_dimension
            column_data = data[:, st : st + dim]
            if column_meta.column_type == DataTypes.CONTINUOUS:
                recovered_column_data = self._reverse_transform_continuous(
                    column_meta, column_data
                )
            else:
                recovered_column_data = self._reverse_transform_categorical(
                    column_meta, column_data
                )

            recovered_column_data_list.append(recovered_column_data)
            column_names.append(column_meta.column_name)
            st += dim

        recovered_data_matrix = np.column_stack(recovered_column_data_list)
        recovered_data = pd.DataFrame(
            recovered_data_matrix, columns=column_names
        ).astype(self._column_raw_dtypes)
        return recovered_data

    def convert_column_name_value_to_id(
        self, column_name: str, value: str
    ) -> Dict[str, int]:
        """Get the ids of the given `column_name`.

        Args:
            column_name (str): Name of the column.
            value (str): Value in the column.

        Returns:
            Dict[str, int]: Dictionary with column and value IDs.
        """
        categorical_counter = 0
        column_id = 0
        for column_meta in self.metadata:
            if column_meta.column_name == column_name:
                break
            if column_meta.column_type == DataTypes.CATEGORICAL:
                categorical_counter += 1

            column_id += 1

        else:
            raise ValueError(
                f"The column_name `{column_name}` doesn't exist in the data."
            )

        ohe = column_meta.transform_function
        # convert the value to a one hot vector via the one hot encoder
        # (has to be via a pd Dataframe)
        data = pd.DataFrame([value], columns=[column_meta.column_name])
        one_hot = ohe.transform(data).to_numpy()[0]
        # normally one element of the vector should be one
        if sum(one_hot) == 0:
            raise ValueError(
                f"The value `{value}` doesn't exist in the column `{column_name}`."
            )

        # returns the index via argmax
        return {
            "categorical_column_id": categorical_counter,
            "column_id": column_id,
            "value_id": int(np.argmax(one_hot)),
        }

    def state_dict(self) -> Dict[str, Any]:
        """Get the state dictionary of the transformer.

        Returns:
            Dict[str, Any]: State dictionary.
        """
        return {
            "output_dimension": self.output_dimension,
            "dataframe": self.dataframe,
            "raw_datatypes": self._column_raw_dtypes,
            "metadata": self.metadata,
            "max_clusters": self._max_clusters,
            "weight_threshold": self._weight_threshold,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load the state dictionary into the transformer.

        Args:
            state_dict (Dict[str, Any]): State dictionary.
        """
        self.output_dimension = state_dict["output_dimension"]
        self.dataframe = state_dict["dataframe"]
        self._column_raw_dtypes = state_dict["raw_datatypes"]
        self.metadata = state_dict["metadata"]
        self._max_clusters = state_dict["max_clusters"]
        self._weight_threshold = state_dict["weight_threshold"]

    def _fit_continuous(self, data: pd.DataFrame) -> ColumnMeta:
        """Train Bayesian GMM for continuous columns.

        Args:
            data (pd.DataFrame): A dataframe containing a column.

        Returns:
            ColumnMeta: Metadata for the column.
        """
        column_name = data.columns[0]
        mean = data[column_name].mean()
        std = data[column_name].std()

        percentiles = np.linspace(0, 1, 100)
        percentile_values = np.quantile(data[column_name], percentiles)

        # keep track of missing values by producing an additional column which says this
        gm = ClusterBasedNormalizer(
            missing_value_generation="from_column",
            max_clusters=min(len(data), self._max_clusters),
            weight_threshold=self._weight_threshold,
        )
        gm.fit(data, column_name)
        num_components = sum(gm.valid_component_indicator)

        return ColumnMeta(
            column_name=column_name,
            column_type=DataTypes.CONTINUOUS,
            transform_function=gm,
            output_dimension=1 + num_components,
            datarepresentations=[
                DataRepresentation(1, DataTypes.FLOAT),
                DataRepresentation(num_components, DataTypes.ONEHOT),
            ],
            original_dtype=(DataTypes.INT if gm._dtype == int else DataTypes.FLOAT),
            percentile_values=percentile_values,
            mean=mean,
            std=std,
        )

    def _fit_categorical(self, data: pd.DataFrame) -> ColumnMeta:
        """Fit one hot encoder for categorical column.

        Args:
            data (pd.DataFrame): A dataframe containing a column.

        Returns:
            ColumnMeta: Metadata for the column.
        """
        column_name = data.columns[0]
        ohe = OneHotEncoder()
        ohe.fit(data, column_name)
        num_categories = len(ohe.dummies)

        return ColumnMeta(
            column_name=column_name,
            column_type=DataTypes.CATEGORICAL,
            original_dtype=DataTypes.STRING,
            transform_function=ohe,
            output_dimension=num_categories,
            datarepresentations=[DataRepresentation(num_categories, DataTypes.ONEHOT)],
            value_frequencies=dict(data[column_name].value_counts() / len(data)),
        )

    def _transform_continuous(
        self, column_meta: ColumnMeta, data: pd.DataFrame
    ) -> np.ndarray:
        """Transform continuous column data.

        Args:
            column_meta (ColumnMeta): Metadata for the column.
            data (pd.DataFrame): Data to be transformed.

        Returns:
            np.ndarray: Transformed data.
        """
        column_name = data.columns[0]
        flattened_column = data[column_name].to_numpy().flatten()
        data = data.assign(**{column_name: flattened_column})
        gm = column_meta.transform_function
        transformed = gm.transform(data)

        #  Converts the transformed data to the appropriate output format.
        #  The first column (ending in '.normalized') stays the same,
        #  but the label encoded column (ending in '.component') is one hot encoded.
        output = np.zeros((len(transformed), column_meta.output_dimension))
        output[:, 0] = transformed[f"{column_name}.normalized"].to_numpy()
        index = transformed[f"{column_name}.component"].to_numpy().astype(int)
        # makes a one hot encoded vector out of the component
        output[np.arange(index.size), index + 1] = 1.0

        return output

    def _transform_categorical(
        self, column_meta: ColumnMeta, data: pd.DataFrame
    ) -> np.ndarray:
        """Transform categorical column data.

        Args:
            column_meta (ColumnMeta): Metadata for the column.
            data (pd.DataFrame): Data to be transformed.

        Returns:
            np.ndarray: Transformed data.
        """
        ohe = column_meta.transform_function
        return ohe.transform(data).to_numpy()

    def _synchronous_transform(
        self, raw_data: pd.DataFrame, metadata: MetaData
    ) -> List[np.ndarray]:
        """Take a Pandas DataFrame and transform columns synchronously.

        Args:
            raw_data (pd.DataFrame): Raw input data.
            metadata (MetaData): Metadata for the columns.

        Returns:
            List[np.ndarray]: List of transformed column data.
        """
        column_data_list = []
        for column_name in raw_data.columns:
            column_meta = metadata[column_name]
            data = raw_data[[column_name]]
            if column_meta.column_type == DataTypes.CONTINUOUS:
                column_data_list.append(self._transform_continuous(column_meta, data))
            else:
                column_data_list.append(self._transform_categorical(column_meta, data))

        return column_data_list

    def _parallel_transform(
        self, raw_data: pd.DataFrame, metadata: MetaData
    ) -> List[np.ndarray]:
        """Take a Pandas DataFrame and transform columns in parallel.

        Args:
            raw_data (pd.DataFrame): Raw input data.
            metadata (MetaData): Metadata for the columns.

        Returns:
            List[np.ndarray]: List of transformed column data.
        """
        processes = []
        for column_name in raw_data.columns:
            column_meta = metadata[column_name]
            data = raw_data[[column_name]]
            process = None
            if column_meta.column_type == DataTypes.CONTINUOUS:
                process = delayed(self._transform_continuous)(column_meta, data)
            else:
                process = delayed(self._transform_categorical)(column_meta, data)
            processes.append(process)

        return Parallel(n_jobs=-1)(processes)

    def _reverse_transform_continuous(
        self, column_meta: ColumnMeta, column_data: Union[np.ndarray, torch.Tensor]
    ) -> pd.DataFrame:
        """Reverse transform continuous column data.

        Args:
            column_meta (ColumnMeta): Metadata for the column.
            column_data (Union[np.ndarray, torch.Tensor]): Transformed column data.

        Returns:
            pd.DataFrame: Reversed column data.
        """
        gm = column_meta.transform_function

        # contruct a new output dataframe of 2 columns with the second being the mode
        data = pd.DataFrame(column_data[:, :2], columns=list(gm.get_output_sdtypes()))
        data[data.columns[1]] = np.argmax(column_data[:, 1:], axis=1)

        return gm.reverse_transform(data)

    def _reverse_transform_categorical(
        self, column_meta: ColumnMeta, column_data: Union[np.ndarray, torch.Tensor]
    ) -> pd.DataFrame:
        """Reverse transform categorical column data.

        Args:
            column_meta (ColumnMeta): Metadata for the column.
            column_data (Union[np.ndarray, torch.Tensor]): Transformed column data.

        Returns:
            pd.DataFrame: Reversed column data.
        """
        ohe = column_meta.transform_function
        data = pd.DataFrame(column_data, columns=list(ohe.get_output_sdtypes()))
        return ohe.reverse_transform(data)[column_meta.column_name]


class SimpleTransformer:
    def __init__(self, metadata: Optional[MetaData] = None) -> None:
        """Create a simple data transformer.

        Args:
            metadata (MetaData): Metadata of a CtganTransformer for the columns.
        """
        self._ctgan_transformer = CtganTransformer()

        if metadata is not None:
            if metadata[0].transform_function is None:
                raise ValueError(
                    "The metadata must be generated by a CtganTransformer instance."
                )

            self._ctgan_transformer.metadata = metadata
            self._ctgan_transformer.output_dimension = (
                metadata.num_transformed_columns()
            )
            self.metadata = adjust_metadata_for_simple_transform(metadata)

    def fit(
        self,
        raw_data: pd.DataFrame,
        categorical_columns: Optional[Iterable[str]] = None,
    ) -> None:

        if self._ctgan_transformer.metadata is None:
            self._ctgan_transformer.fit(raw_data, categorical_columns)
        self._ctgan_transformer.output_dimension = (
            self._ctgan_transformer.metadata.num_transformed_columns()
        )
        self.metadata = adjust_metadata_for_simple_transform(
            self._ctgan_transformer.metadata
        )

    def transform(self, raw_data: Union[pd.Series, pd.DataFrame]) -> torch.Tensor:
        return simple_transform(raw_data, self.metadata)

    def reverse_transform(self, transformed_data: torch.Tensor) -> pd.DataFrame:
        return reverse_simple_transform(transformed_data, self.metadata)

    def convert_column_name_value_to_id(
        self, column_name: str, value: str
    ) -> Dict[str, int]:
        """Get the ids of the given `column_name`.

        Args:
            column_name (str): Name of the column.
            value (str): Value in the column.

        Returns:
            Dict[str, int]: Dictionary with column and value IDs.
        """
        categorical_counter = 0
        column_id = 0
        for column_meta in self.metadata:
            if column_meta.column_name == column_name:
                break
            if column_meta.column_type == DataTypes.CATEGORICAL:
                categorical_counter += 1

            column_id += 1

        else:
            raise ValueError(
                f"The column_name `{column_name}` doesn't exist in the data."
            )

        value_id = list(column_meta.value_frequencies).index(value)

        # returns the index via argmax
        return {
            "categorical_column_id": categorical_counter,
            "column_id": column_id,
            "value_id": value_id,
        }

    def state_dict(self) -> Dict[str, Any]:
        """Get the state dictionary of the transformer.

        Returns:
            Dict[str, Any]: State dictionary.
        """
        return {
            "ctgan_transformer": self._ctgan_transformer.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load the state dictionary into the transformer.

        Args:
            state_dict (Dict[str, Any]): State dictionary.
        """
        self._ctgan_transformer.load_state_dict(state_dict["ctgan_transformer"])
        self.metadata = adjust_metadata_for_simple_transform(
            self._ctgan_transformer.metadata
        )


def simple_transform(
    raw_data: Union[pd.Series, pd.DataFrame], metadata: MetaData
) -> torch.Tensor:
    """Simpler transformation of raw data.

    Args:
        raw_data (Union[pd.Series, pd.DataFrame]): Raw input data.
        metadata (MetaData): Metadata for the columns.

    Returns:
        torch.Tensor: Transformed tensor data.
    """
    column_data_list = []

    if isinstance(raw_data, pd.Series):
        raw_data = raw_data.to_frame()

    for column_name in raw_data.columns:
        column_meta = metadata[column_name]
        data = raw_data[column_name]

        if column_meta.column_type == DataTypes.CONTINUOUS:
            normalized_data = (data.values - column_meta.mean) / column_meta.std
            column_data_list.append(normalized_data.reshape(-1, 1))
        else:
            # value frequencies has unique values as keys
            indices = data.apply(lambda x: list(column_meta.value_frequencies).index(x))
            one_hot = np.zeros((len(data), len(column_meta.value_frequencies)))
            one_hot[np.arange(len(data)), indices] = 1
            column_data_list.append(one_hot)

    concatenated_data = np.concatenate(column_data_list, axis=1).astype(np.float32)
    return torch.tensor(concatenated_data, dtype=torch.float32)


def reverse_simple_transform(
    transformed_data: torch.Tensor, metadata: MetaData
) -> pd.DataFrame:
    """Reverse a simpler transformation of data.

    Args:
        transformed_data (torch.Tensor): Transformed tensor data.
        metadata (MetaData): Metadata for the columns.

    Returns:
        pd.DataFrame: Reversed raw data.
    """
    transformed_data_np = transformed_data.numpy()
    reversed_data = {}
    col_idx = 0

    for column_meta in metadata:
        column_name = column_meta.column_name
        if column_meta.column_type == "continuous":
            column_data = transformed_data_np[:, col_idx]
            reversed_col = column_data * column_meta.std + column_meta.mean
            if column_meta.original_dtype == DataTypes.INT:
                reversed_col = reversed_col.round().astype(int)

            reversed_data[column_name] = reversed_col
            col_idx += 1
        else:
            num_categories = len(list(column_meta.value_frequencies.keys()))
            one_hot_data = transformed_data_np[:, col_idx : col_idx + num_categories]
            original_data_indices = np.argmax(one_hot_data, axis=1)
            reversed_data[column_name] = [
                list(column_meta.value_frequencies.keys())[idx]
                for idx in original_data_indices
            ]
            col_idx += num_categories

    return pd.DataFrame(reversed_data)


def adjust_metadata_for_simple_transform(metadata: MetaData) -> MetaData:
    """Serves purely as a simple hack to reuse most code.

    Args:
        metadata (MetaData): Metadata for the columns.

    Returns:
        MetaData: Adjusted metadata for simple transform.
    """
    meta_list = []
    for column_meta in metadata:
        name = column_meta.column_name
        type = column_meta.column_type
        value_freqs = column_meta.value_frequencies
        perc_values = column_meta.percentile_values
        mean = column_meta.mean
        std = column_meta.std

        if type == DataTypes.CATEGORICAL:
            output_dim = len(column_meta.value_frequencies.keys())
            datarepresentation = [DataRepresentation(output_dim, DataTypes.ONEHOT)]
            dtype = DataTypes.STRING
        else:
            output_dim = 1
            datarepresentation = [DataRepresentation(output_dim, DataTypes.FLOAT)]
            dtype = (
                DataTypes.INT
                if column_meta.transform_function._dtype == int
                else DataTypes.FLOAT
            )

        meta_list.append(
            ColumnMeta(
                column_name=name,
                column_type=type,
                transform_function=None,
                output_dimension=output_dim,
                datarepresentations=datarepresentation,
                original_dtype=dtype,
                value_frequencies=value_freqs,
                percentile_values=perc_values,
                mean=mean,
                std=std,
            )
        )
    return MetaData(meta_list)
