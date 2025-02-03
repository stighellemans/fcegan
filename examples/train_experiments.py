from itertools import product
from pathlib import Path
from typing import Dict, Sequence, Union

import pandas as pd

from counterfactual_package.engines.training import train_fcegan
from counterfactual_package.models.classifiers import Classifier
from counterfactual_package.utils.utils import (
    get_checkpoint,
    get_model,
    load_json_file,
)

ABBREVIATIONS = {
    "counterfactual_disc_influence": "cf_disc_infl",
    "original_disc_influence": "og_disc_infl",
    "reconstruction_divergence_influence": "reconst_div_infl",
    "cf_divergence_influence": "cf_div_infl",
    "classifier_influence": "class_infl",
    "discriminator_lr": "disc_lr",
    "generator_lr": "gen_lr",
    "cf_dropout_range": "cf_drop",
}


def iterate_hyperparameters(
    expand=False, **hyperparameters: Sequence[Union[int, float]]
):
    """
    Iterate over combinations of hyperparameters.

    Parameters:
    expand (bool): If False, iterate over hyperparameters in a synchronized manner.
                   If True, iterate over all possible combinations of hyperparameters.
    **hyperparameters (Dict[str, Sequence[Union[int, float]]]): Hyperparameters to\
         iterate over.

    Yields:
    dict: A dictionary of hyperparameter names and their corresponding values.

    Raises:
    ValueError: If `expand` is False and the sequences of hyperparameters do not have\
         the same length.
    """
    if not expand:
        names = list(hyperparameters.keys())
        values = list(hyperparameters.values())

        # Check if all sequences have the same length
        length = len(values[0])
        if not all(len(v) == length for v in values):
            raise ValueError("All sequences must have the same length")

        for i in range(length):
            yield {names[j]: values[j][i] for j in range(len(names))}
    else:
        keys = hyperparameters.keys()
        for combination in product(*hyperparameters.values()):
            yield dict(zip(keys, combination))


def make_file_name(
    hyperparameters: Dict[str, Union[int, float]], selection_order: Sequence[str]
) -> str:
    return ",".join(f"{ABBREVIATIONS[k]}={hyperparameters[k]}" for k in selection_order)


if __name__ == "__main__":
    output_dir = Path("experiments")
    data_dir = Path("data")

    config = load_json_file(output_dir / "config.json")

    hyperparameters = {
        "adult": {
            "reconstruction_divergence_influence": [0, 0, 0, 0, 0, 0],
            "cf_divergence_influence": [1, 0, 10, 100, 5, 50],
            "classifier_influence": [1, 0, 1, 1, 0, 0],
        },
        "heart_disease": {
            "reconstruction_divergence_influence": [0, 0, 0, 0, 0, 0],
            "cf_divergence_influence": [1, 0, 10, 100, 5, 50],
            "classifier_influence": [1, 0, 1, 1, 0, 0],
        },
        "diabetes": {
            "reconstruction_divergence_influence": [0, 0, 0, 0, 0, 0],
            "cf_divergence_influence": [1, 0, 10, 100, 5, 50],
            "classifier_influence": [1, 0, 1, 1, 0, 0],
        },
        "employees": {
            "reconstruction_divergence_influence": [0, 0, 0, 0, 0, 0],
            "cf_divergence_influence": [1, 0, 10, 100, 5, 50],
            "classifier_influence": [1, 0, 1, 1, 0, 0],
        },
    }


    file_name_params = [
        "counterfactual_disc_influence",
        "original_disc_influence",
        "reconstruction_divergence_influence",
        "cf_divergence_influence",
        "classifier_influence",
        "discriminator_lr",
        "generator_lr",
        "cf_dropout_range",
    ]

    fcegan_types = [
        ("simple", "fcegan"),
        ("simple", "no_template_fcegan"),
        # ("ctgan", "fcegan"),
        # ("ctgan", "no_template_fcegan"),
    ]

    dataset_names = ["adult", "heart_disease", "diabetes", "employees"]
    trials = [1, 2, 3, 4, 5]

    for trial in trials:
        print("trial", trial)
        for transform_name, fcegan_type in fcegan_types:
            if transform_name == "simple":
                fcegan_name = "simple_" + fcegan_type
            else:
                fcegan_name = fcegan_type

            for dataset_name in dataset_names:
                output_path = output_dir / dataset_name

                fcegan_config = config[dataset_name]["fcegan"]
                fcegan_config["target_name"] = (
                    config[dataset_name]["target_name"] + "_pred"
                )
                fcegan_config["transformer_path"] = str(
                    get_model(
                        output_dir / dataset_name,
                        f"{transform_name}_prediction_transformer",
                        ".pth",
                    )
                )
                fcegan_config["eval_ctgan_path"] = str(
                    get_model(
                        output_dir / dataset_name,
                        f"{transform_name}_prediction_eval_ctgan",
                        ".pth",
                    )
                )

                # Load the classifier
                fcegan_config["classifier_name"] = (
                    f"{transform_name}_transform_classifier"
                )
                classifier_config = load_json_file(
                    get_model(
                        output_dir / dataset_name,
                        fcegan_config["classifier_name"],
                        "config.json",
                    )
                )
                classifier = Classifier(**classifier_config)
                classifier.load_state_dict(
                    get_checkpoint(
                        get_model(
                            output_path,
                            f"{transform_name}_transform_classifier",
                            ".pth",
                        )
                    )["model_state_dict"]
                )

                # Load data
                fcegan_config["train_data"] = (
                    f"{dataset_name}_train_{transform_name}_predicted.csv"
                )
                fcegan_config["test_data"] = (
                    f"{dataset_name}_val_{transform_name}_predicted.csv"
                )
                train = pd.read_csv(
                    data_dir / dataset_name / fcegan_config["train_data"]
                )
                test = pd.read_csv(data_dir / dataset_name / fcegan_config["test_data"])

                if fcegan_type == "no_template_fcegan":
                    # Not resetting immutable features during training -> no template
                    fcegan_config["cf_dropout_range"] = (1.0, 1.0)
                else:
                    fcegan_config["cf_dropout_range"] = (0.1, 1.0)

                dataset_hyperparams = hyperparameters[dataset_name]
                for exp_hyperparams in iterate_hyperparameters(**dataset_hyperparams):
                    fcegan_config.update(exp_hyperparams)
                    save_path = (
                        output_dir
                        / dataset_name
                        / "fcegan"
                        / fcegan_name
                        / make_file_name(
                            fcegan_config, selection_order=file_name_params
                        )
                        / f"trial{trial}"
                    )

                    print(save_path)

                    train_fcegan(
                        train,
                        test,
                        classifier,
                        fcegan_config,
                        save_path,
                        fcegan_type=fcegan_name,
                        resume=False,
                        only_best=True,
                        saving_each_x_epoch=1,
                    )
