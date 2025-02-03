"""
This module provides utilities for device management, file operations, and configuration
checks, including checking for duplicates in sequences, handling JSON files, managing
PyTorch optimizers, and ensuring consistency in configurations.
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, Sequence, Union

import torch
import torch.optim as optim

logger = logging.getLogger(__name__)


def has_duplicates(seq: Sequence[Any]) -> bool:
    """
    Checks if the given sequence contains any duplicates.

    Args:
        seq (Sequence[Any]): The sequence to check for duplicates.

    Returns:
        bool: True if duplicates are found, False otherwise.
    """
    return len(seq) != len(set(seq))


def check_available_device() -> torch.device:
    """Device-agnostic checking the best available device (apple silicon supported)"""
    try:
        # Attempt to use MPS (Apple Silicon support) if available
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    except AttributeError:
        # Fallback to CUDA or CPU if MPS is not supported
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    return device


def move_optimizer_to_device(optimizer: optim.Optimizer, device: torch.device):
    """
    Moves all states of an optimizer to a specified device.

    Args:
        optimizer (optim.Optimizer): The optimizer whose states need to be moved.
        device (torch.device): The target device.
    """
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


def load_json_file(file_path: Union[str, Path]) -> dict:
    """
    Loads a JSON file and returns its content as a dictionary.

    Args:
        file_path (Union[str, Path]): Path to the JSON file.

    Returns:
        dict: The content of the JSON file.
    """
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def write_json_file(file_path: Union[str, Path], output: Any):
    """
    Writes a dictionary or any serializable object to a JSON file.

    Args:
        file_path (Union[str, Path]): Path to the JSON file.
        output (Any): The data to be written to the JSON file.
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(output, f, indent=4)


def remove_folder(path: Union[str, Path]):
    """
    Removes a folder and all its contents.

    Args:
        path (Union[str, Path]): The path to the folder to be removed.
    """
    Path(path).mkdir(parents=True, exist_ok=True)
    shutil.rmtree(path)


def same_config(d1: Dict[str, Any], d2: Dict[str, Any], ignore: Sequence[str] = ()):
    """
    Compares two dictionaries to check if they have the same configuration,
    optionally ignoring specified keys.

    Args:
        d1 (Dict[str, Any]): The first dictionary.
        d2 (Dict[str, Any]): The second dictionary.
        ignore (Sequence[str]): A sequence of keys to ignore during comparison.

    Returns:
        bool: True if the dictionaries have the same configuration, False otherwise.
    """

    def normalize(value: Any) -> Any:
        if isinstance(value, Iterable):
            # Convert all iterables to lists for comparison
            return list(value)
        return value

    if d1.keys() != d2.keys():
        return False

    return all(
        [normalize(d1[hp]) == normalize(d2[hp]) for hp in d1.keys() if hp not in ignore]
    )


def proceed_if_filedir_exists(save_path: Union[str, Path]):
    """
    Checks if a file or directory exists and prompts the user for action.

    Args:
        save_path (Union[str, Path]): The path to check.

    Returns:
        bool: True if the program should proceed, False otherwise.
    """
    save_path = Path(save_path)
    if save_path.exists():
        ask = input(
            f"Filedir '{save_path.name}' already exists. Overwrite (y/n)"
        ).lower()
        if ask == "y":
            print(
                "Old folder  will be overwritten.",
                "The program is continued.",
            )
            remove_folder(save_path)
            return True
        else:
            print("The old folder will not be overwritten. The program is stopped.")
            return False
    else:
        return True


def search_models(
    dir: Union[str, Path], *search_args, **search_args_dict
) -> Generator[Path, None, None]:
    """
    Recursively searches for files with a specific suffix in a directory and\
          its subdirectories,
    and filters them based on additional search arguments.

    Args:
        dir (Union[str, Path]): The directory to search in.
        *search_args: Additional string arguments that must be present in the file path.
        **search_args_dict: Additional key-value arguments that must be present in the\
              file path in the format "key=value".

    Yields:
        Path: Paths of files that match the search criteria.

    Raises:
        FileNotFoundError: If the specified directory does not exist.
        NotADirectoryError: If the specified path is not a directory.
    """
    dir = Path(dir)
    if not dir.exists():
        raise FileNotFoundError(f"Directory {dir} does not exist")
    if not dir.is_dir():
        raise NotADirectoryError(f"{dir} is not a directory")
    for file in dir.iterdir():
        if file.is_dir():
            yield from search_models(file, *search_args, **search_args_dict)
        if all(arg in str(file) for arg in search_args) and all(
            f"{k}={v}" in str(file) for k, v in search_args_dict.items()
        ):
            yield file


def get_model(dir: Union[str, Path], *search_args, **search_args_dict) -> Path:
    """
    Retrieve a single model file from a directory based on the given suffix and\
          search arguments.
    Args:
        dir (Union[str, Path]): The directory to search for the model file.
        *search_args: Additional positional arguments to pass to the search function.
        **search_args_dict: Additional keyword arguments to pass to the search function.
    Returns:
        The path to the single model file found.
    Raises:
        FileNotFoundError: If no model files are found in the directory with the given\
              suffix and search arguments.
        ValueError: If multiple model files are found in the directory with the given\
              suffix and search arguments.
    """

    models = list(search_models(dir, *search_args, **search_args_dict))
    if len(models) == 0:
        raise FileNotFoundError(
            f"No models found in {dir} with"
            f" search arguments {search_args} and {search_args_dict}"
        )
    elif len(models) > 1:
        raise ValueError(
            f"Multiple models found in {dir} with"
            f" search arguments {search_args} and {search_args_dict}"
        )
    return models[0]


def get_checkpoint(model_path: Union[str, Path]) -> dict:
    """
    Loads the checkpoint dictionary of a model from a given file path.

    Args:
        model_path (Union[str, Path]): The path to the file containing the model\
              checkpoint.

    Returns:
        dict: The checkpoint dictionary of the model.
    """
    return torch.load(model_path)["checkpoint"]
