"""
DataSampler module for sampling conditional vectors and corresponding data for CTGAN.
"""

from typing import Any, Dict, Optional, Tuple

import torch

from .metadata import DataTypes, MetaData


class DataSampler:
    """DataSampler samples the conditional vector and corresponding data for CTGAN."""

    def fit(self, data: torch.Tensor, metadata: MetaData, log_frequency: bool = True):
        """Fit the DataSampler with the given data and metadata.

        Args:
            data (torch.Tensor): The input data tensor.
            metadata (MetaData): The metadata for the data.
            log_frequency (bool): Whether to use log frequency for category \
                probabilities.
        """
        self._data = data

        categorical_dimensions = [
            colum_meta.output_dimension
            for colum_meta in metadata
            if colum_meta.column_type == DataTypes.CATEGORICAL
        ]

        n_cat_columns = len(categorical_dimensions)
        self.n_cat_columns = n_cat_columns
        self.total_n_categories = sum(categorical_dimensions)

        # Store the row id for each category in each categorical column.
        # For example _rid_by_cat_cols[a][b] is a list of all rows with the
        # a-th categorical column equal value b.
        self.row_id_by_cat_cols = []
        current_input_id = 0
        for column_meta in metadata:
            dim = column_meta.output_dimension
            if column_meta.column_type != DataTypes.CATEGORICAL:
                current_input_id += dim
            else:
                end = current_input_id + dim
                row_id_by_cat = []
                for value_id in range(dim):
                    # check certain category has certain value (1 if categorical value)
                    row_id_by_cat.append(
                        torch.nonzero(data[:, current_input_id + value_id]).squeeze(1)
                    )
                self.row_id_by_cat_cols.append(row_id_by_cat)
                current_input_id = end

        # check if it has completely gone over all columns
        assert current_input_id == data.shape[1]

        max_n_categories = max(categorical_dimensions, default=0)

        self.categorical_column_cond_start_id = torch.zeros(
            n_cat_columns, dtype=torch.int32
        )
        self.categorical_column_n_values = torch.zeros(n_cat_columns, dtype=torch.int32)
        # make probability matrix of (num categorical columns, max_n_categories)
        self.categorical_column_value_probs = torch.zeros(
            (n_cat_columns, max_n_categories), dtype=torch.float32
        )

        # calculate total number of categories after transformation into one-hot vectors
        current_input_id = 0
        category_id = 0
        current_cond_vector_id = 0
        for colum_meta in metadata:
            dim = colum_meta.output_dimension
            if colum_meta.column_type != DataTypes.CATEGORICAL:
                current_input_id += dim
            else:
                end = current_input_id + dim
                # calculate how much of the samples has a certain category
                # if needed, calculate the log frequency: +1 ensures numerical stability
                category_freq = torch.sum(data[:, current_input_id:end], dim=0)
                if log_frequency:
                    category_freq = torch.log(category_freq + 1)

                # calculate the category probabilities (divided by total)
                category_prob = category_freq / torch.sum(category_freq)
                # put the probs into the matrix (specified for the number of categories)
                self.categorical_column_value_probs[category_id, :dim] = category_prob
                # put into a separate 1d array: the index of the original data
                # for the row index in the probability matrix
                self.categorical_column_cond_start_id[category_id] = (
                    current_cond_vector_id
                )
                # put into a separate 1d array: the number of categories
                # per categorical column for the rowindex in the prob matrix
                self.categorical_column_n_values[category_id] = dim
                current_cond_vector_id += dim
                category_id += 1
                current_input_id = end

    def _choose_random_cat_value_ids(
        self, categorical_column_ids: torch.Tensor
    ) -> torch.Tensor:
        """Choose random category value IDs based on the probabilities."""
        probs = self.categorical_column_value_probs[categorical_column_ids]
        num_ids = probs.shape[0]
        random = torch.rand(num_ids, 1)
        # calculate the cumulative sum over the row, if one exceeds the value of random
        # -> give the first value using argmax
        # thereby gives random category ids for a particular categorical column
        return (torch.cumsum(probs, dim=1) > random).to(dtype=torch.int64).argmax(dim=1)

    def sample_conditional_vectors(
        self, batch_size: int
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Generates conditional vectors for training.

        Args:
            batch_size (int): The number of samples to generate.

        Returns:
            Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
                Conditional vectors, mask vectors, categorical column IDs,
                and category value IDs.
        """
        # if no categorical colums, return None
        if self.n_cat_columns == 0:
            return None

        # choose randomly for each batch a categorical column
        categorical_column_ids = torch.randint(self.n_cat_columns, (batch_size,))
        category_value_ids = self._choose_random_cat_value_ids(categorical_column_ids)
        # get category index in original data shape
        cat_value_cond_ids = (
            self.categorical_column_cond_start_id[categorical_column_ids]
            + category_value_ids
        )

        mask_vectors = torch.zeros(
            (batch_size, self.n_cat_columns), dtype=torch.float32
        )
        mask_vectors[torch.arange(batch_size), categorical_column_ids] = 1

        conditional_vectors = torch.zeros(
            (batch_size, self.total_n_categories), dtype=torch.float32
        )
        conditional_vectors[torch.arange(batch_size), cat_value_cond_ids] = 1

        return (
            conditional_vectors,
            mask_vectors,
            categorical_column_ids,
            category_value_ids,
        )

    def get_data_size(self) -> int:
        """Get the size of the data.

        Returns:
            int: The number of rows in the data.
        """
        return len(self._data)

    def sample_original_cond_vectors(self, batch_size: int) -> Optional[torch.Tensor]:
        """Generate conditional vectors using the original frequency.

        Args:
            batch_size (int): The number of samples to generate.

        Returns:
            Optional[torch.Tensor]: The generated conditional vectors.
        """
        if self.n_cat_columns == 0:
            return None

        conditional_vectors = torch.zeros(
            (batch_size, self.total_n_categories), dtype=torch.float32
        )

        for i in range(batch_size):
            row_idx = torch.randint(0, len(self._data), (1,))
            col_idx = torch.randint(0, self.n_cat_columns, (1,))

            # get from sample the value id of chosen column
            start_cat_id = self.categorical_column_cond_start_id[col_idx]
            end_id = start_cat_id + self.categorical_column_n_values[col_idx]
            value_id = torch.argmax(self._data[row_idx, start_cat_id:end_id])

            # set the specific category of a categorical column to 1
            conditional_vectors[i, start_cat_id + value_id] = 1

        return conditional_vectors

    def sample_data(
        self,
        n: Optional[int] = None,
        cat_col_ids: Optional[torch.Tensor] = None,
        value_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample data from original training data satisfying the sampled conditional \
            vector.

        Args:
            n (Optional[int]): The number of samples to generate.
            cat_col_ids (Optional[torch.Tensor]): The categorical column IDs to satisfy.
            value_ids (Optional[torch.Tensor]): The category value IDs to satisfy.

        Returns:
            torch.Tensor: The sampled data.
        """
        # if no columns are specified, return a random number
        # of rows of the data (size n)
        if cat_col_ids is None:
            idxs = torch.randint(len(self._data), (n,))
        else:
            idxs = []
            # if categorical column + specific category is given
            # choose from list of rows that have this specific
            for c, o in zip(cat_col_ids, value_ids):
                candidates = self.row_id_by_cat_cols[c][o]
                random_idx = torch.randint(len(candidates), (1,)).item()
                idxs.append(candidates[random_idx])

        return self._data[idxs]

    def dim_conditional_vector(self) -> int:
        """Return the total number of categories.

        Returns:
            int: The total number of categories.
        """
        return self.total_n_categories

    def construct_conditional_vector(
        self, cat_column_id: int, value_id: int, batch_size: int
    ) -> torch.Tensor:
        """Construct a batch of conditional vectors.

        Args:
            cat_column_id (int): The categorical column ID.
            value_id (int): The value ID within the categorical column.
            batch_size (int): The number of samples to generate.

        Returns:
            torch.Tensor: The constructed conditional vectors.
        """
        conditional_vectors = torch.zeros(
            (batch_size, self.total_n_categories), dtype=torch.float32
        )
        id_ = int(self.categorical_column_cond_start_id[cat_column_id])
        # adjust the index to get a specific category
        id_ += value_id
        # set this specific category of a categorical column to 1
        conditional_vectors[:, id_] = 1
        return conditional_vectors

    def state_dict(self) -> Dict[str, Any]:
        """Get the state dictionary of the DataSampler.

        Returns:
            Dict[str, Any]: The state dictionary.
        """
        return {
            "categorical_column_cond_start_id": self.categorical_column_cond_start_id,
            "categorical_column_n_values": self.categorical_column_n_values,
            "categorical_column_value_probs": self.categorical_column_value_probs,
            "data": self._data,
            "n_cat_columns": self.n_cat_columns,
            "row_id_by_cat_cols": self.row_id_by_cat_cols,
            "total_n_categories": self.total_n_categories,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load the state dictionary into the DataSampler.

        Args:
            state_dict (Dict[str, Any]): The state dictionary to load.
        """
        self.categorical_column_cond_start_id = state_dict[
            "categorical_column_cond_start_id"
        ]
        self.categorical_column_n_values = state_dict["categorical_column_n_values"]
        self.categorical_column_value_probs = state_dict[
            "categorical_column_value_probs"
        ]
        self._data = state_dict["data"]
        self.n_cat_columns = state_dict["n_cat_columns"]
        self.row_id_by_cat_cols = state_dict["row_id_by_cat_cols"]
        self.total_n_categories = state_dict["total_n_categories"]
