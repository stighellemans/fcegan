import pickle
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from counterfactual_package.data.datasets import (
    switch_adult_target,
    switch_diabetes_target,
    switch_employees_target,
    switch_heart_disease_target,
)
from counterfactual_package.engines.counterfactual_synthesizer import (
    FCEGAN,
    NoTemplateFCEGAN,
    SimpleFCEGAN,
    SimpleNoTemplateFCEGAN,
)
from counterfactual_package.engines.ctgan_synthesizer import (
    CtganSynthesizer,
    SimpleCtganSynthesizer,
)
from counterfactual_package.models.classifiers import Classifier
from counterfactual_package.utils.utils import (
    get_checkpoint,
    get_model,
    load_json_file,
)
from flexibility import (
    calculate_flexibility,
    calculate_flexibility_optimizer,
    calculate_flexibility_random_generator,
)

output_dir = Path("./experiments")
data_dir = Path("./data")
results_dir = output_dir / "results"

dataset_names = ["adult", "heart_disease", "diabetes", "employees"]

config = load_json_file(output_dir / "config.json")

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

roundings_dicts = {
    "adult": {
        "Age": 0,
        "Education-Num": 0,
        "Capital Gain": 0,
        "Capital Loss": 0,
        "Hours per week": 0,
    },
    "heart_disease": {
        "Height_(cm)": 0,
        "Weight_(kg)": 2,
        "BMI": 2,
        "Alcohol_Consumption": 0,
        "Fruit_Consumption": 0,
        "Green_Vegetables_Consumption": 0,
        "FriedPotato_Consumption": 0,
    },
    "diabetes": {
        "Body_Mass_Index": 0,
        "Mental_Health": 0,
        "Physical_Health": 0,
        "Difficulty_Walking": 0,
    },
    "employees": {
        "Age": 0,
        "Years_at_Company": 0,
        "Monthly_Income": 0,
        "Number_of_Promotions": 0,
        "Distance_from_Home": 0,
        "Number_of_Dependents": 0,
        "Company_Tenure": 0,
    },
}

basic_hyperparams = {
    "counterfactual_disc_influence": 0.5,
    "original_disc_influence": 0.5,
    "cf_divergence_influence": 0,
    "discriminator_lr": 0.0002,
    "generator_lr": 0.0002,
}

fcegan_models = {
    "clas_cfgan_reconst_div=0.5": {
        "reconstruction_divergence_influence": {
            "adult": 0.5,
        },
        "classifier_influence": 1,
        "cf_dropout_range": (0.1, 1.0),
    },
    "clas_cfgan_reconst_div=5": {
        "reconstruction_divergence_influence": {
            "adult": 5,
        },
        "classifier_influence": 1,
        "cf_dropout_range": (0.1, 1.0),
    },
    "black_box_cfgan_reconst_div=1": {
        "reconstruction_divergence_influence": {
            "adult": 1,
        },
        "classifier_influence": 0,
        "cf_dropout_range": (0.1, 1.0),
    },
    "black_box_cfgan_reconst_div=10": {
        "reconstruction_divergence_influence": {
            "adult": 10,
        },
        "classifier_influence": 0,
        "cf_dropout_range": (0.1, 1.0),
    },
    "clas_cfgan_cf_div=0.5": {
        "reconstruction_divergence_influence": {
            "adult": 0,
        },
        "cf_divergence_influence": 0.5,
        "classifier_influence": 1,
        "cf_dropout_range": (0.1, 1.0),
    },
    "clas_cfgan_cf_div=1": {
        "reconstruction_divergence_influence": {
            "adult": 0,
        },
        "cf_divergence_influence": 1,
        "classifier_influence": 1,
        "cf_dropout_range": (0.1, 1.0),
    },
    "clas_cfgan_cf_div=5": {
        "reconstruction_divergence_influence": {
            "adult": 0,
        },
        "cf_divergence_influence": 5,
        "classifier_influence": 1,
        "cf_dropout_range": (0.1, 1.0),
    },
    "clas_cfgan_cf_div=10": {
        "reconstruction_divergence_influence": {
            "adult": 0,
            "heart_disease": 0,
            "diabetes": 0,
            "employees": 0,
        },
        "cf_divergence_influence": 10,
        "classifier_influence": 1,
        "cf_dropout_range": (0.1, 1.0),
    },
    "clas_cfgan_cf_div=100": {
        "reconstruction_divergence_influence": {
            "adult": 0,
            "heart_disease": 0,
            "diabetes": 0,
            "employees": 0,
        },
        "cf_divergence_influence": 100,
        "classifier_influence": 1,
        "cf_dropout_range": (0.1, 1.0),
    },
    "black_box_cfgan_cf_div=0.5": {
        "reconstruction_divergence_influence": {
            "adult": 0,
        },
        "cf_divergence_influence": 0.5,
        "classifier_influence": 0,
        "cf_dropout_range": (0.1, 1.0),
    },
    "black_box_cfgan_cf_div=1": {
        "reconstruction_divergence_influence": {
            "adult": 0,
        },
        "cf_divergence_influence": 1,
        "classifier_influence": 0,
        "cf_dropout_range": (0.1, 1.0),
    },
    "black_box_cfgan_cf_div=5": {
        "reconstruction_divergence_influence": {
            "adult": 0,
            "heart_disease": 0,
            "diabetes": 0,
            "employees": 0,
        },
        "cf_divergence_influence": 5,
        "classifier_influence": 0,
        "cf_dropout_range": (0.1, 1.0),
    },
    "black_box_cfgan_cf_div=50": {
        "reconstruction_divergence_influence": {
            "adult": 0,
            "heart_disease": 0,
            "diabetes": 0,
            "employees": 0,
        },
        "cf_divergence_influence": 50,
        "classifier_influence": 0,
        "cf_dropout_range": (0.1, 1.0),
    },
    "black_box_cfgan_cf_div=10": {
        "reconstruction_divergence_influence": {
            "adult": 0,
        },
        "cf_divergence_influence": 10,
        "classifier_influence": 0,
        "cf_dropout_range": (0.1, 1.0),
    },
    "clas_cfgan_only_cf_disc": {
        "counterfactual_disc_influence": 1,
        "original_disc_influence": 0,
        "reconstruction_divergence_influence": {
            "adult": 0,
            "employees": 0,
        },
        "classifier_influence": 1,
        "cf_dropout_range": (0.1, 1.0),
    },
    "clas_cfgan_only_og_disc": {
        "counterfactual_disc_influence": 0,
        "original_disc_influence": 1,
        "reconstruction_divergence_influence": {
            "adult": 0,
            "employees": 0,
        },
        "classifier_influence": 1,
        "cf_dropout_range": (0.1, 1.0),
    },
    "black_box_only_cf_disc": {
        "counterfactual_disc_influence": 1,
        "original_disc_influence": 0,
        "reconstruction_divergence_influence": {
            "adult": 0,
            "employees": 0,
        },
        "classifier_influence": 0,
        "cf_dropout_range": (0.1, 1.0),
    },
    "black_box_only_og_disc": {
        "counterfactual_disc_influence": 0,
        "original_disc_influence": 1,
        "reconstruction_divergence_influence": {
            "adult": 0,
            "employees": 0,
        },
        "classifier_influence": 0,
        "cf_dropout_range": (0.1, 1.0),
    },
        "class_no_template_cfgan": {
        "reconstruction_divergence_influence": {
            dataset_name: 0 for dataset_name in dataset_names
        },
        "classifier_influence": 1,
        "cf_dropout_range": (1.0, 1.0),
    },
    "black_box_no_template_cfgan": {
        "reconstruction_divergence_influence": {
            dataset_name: 0 for dataset_name in dataset_names
        },
        "classifier_influence": 0,
        "cf_dropout_range": (1.0, 1.0),
    },
}


optimizations = {
    "template_optimization_no_div": {
        "divergence_influence": 0,
        "num_steps": 30,
        "template_guided": True,
    },
    "template_optimization_with_div": {
        "divergence_influence": 1,
        "num_steps": 30,
        "template_guided": True,
    },
    "default_optimization_no_div": {
        "divergence_influence": 0,
        "num_steps": 30,
        "template_guided": False,
    },
}

flexibilities = {
    transform_name: {dataset_name: {} for dataset_name in dataset_names}
    for transform_name in ["simple", "ctgan"]
}

trials = [1, 2, 3, 4, 5]

for trial in trials:
    # for transform_name in ["simple", "ctgan"]:
    for transform_name in ["simple"]:
        print("Transform:", transform_name)
        for dataset_name in dataset_names:
            print("Dataset:", dataset_name)

            # Load data
            test = pd.read_csv(
                data_dir
                / dataset_name
                / f"{dataset_name}_test_{transform_name}_predicted.csv"
            )
            rounding_dict = roundings_dicts[dataset_name]

            # Load classifier
            classifier_config = load_json_file(
                get_model(
                    output_dir / dataset_name,
                    transform_name,
                    "classifier",
                    "config.json",
                )
            )
            classifier = Classifier(**classifier_config)
            classifier.load_state_dict(
                get_checkpoint(
                    get_model(
                        output_dir / dataset_name,
                        transform_name,
                        "classifier",
                        ".pth",
                    )
                )["model_state_dict"]
            )

            # Load eval CTGAN
            eval_ctgan = CtganSynthesizer()
            eval_ctgan.load_state_dict(
                torch.load(
                    get_model(
                        output_dir / dataset_name,
                        f"{transform_name}_prediction_eval_ctgan",
                        ".pth",
                    )
                )
            )

            switch_target_fn = globals().get(f"switch_{dataset_name}_target")
            # make batch size a multiple of 1000 and at most 40000
            batch_size = 40000 if len(test) > 40000 else len(test) // 1000 * 1000

            dataset_flexibilities = {}
            for fcegan_name, hyperparameters in tqdm(
                fcegan_models.items(), total=len(fcegan_models)
            ):

                if "no_template" in fcegan_name and transform_name == "simple":
                    fcegan = SimpleNoTemplateFCEGAN()
                elif "no_template" in fcegan_name and transform_name == "ctgan":
                    fcegan = NoTemplateFCEGAN()
                elif transform_name == "simple":
                    fcegan = SimpleFCEGAN()
                else:
                    fcegan = FCEGAN()

                hyperparameters = {**basic_hyperparams, **hyperparameters}
                hyperparameters["reconstruction_divergence_influence"] = (
                    hyperparameters["reconstruction_divergence_influence"][dataset_name]
                )
                search_params = {
                    ABBREVIATIONS[key]: f"{value},"
                    for key, value in hyperparameters.items()
                    if key in ABBREVIATIONS and key != "cf_dropout_range"
                }
                search_params["cf_drop"] = hyperparameters["cf_dropout_range"]
                extra_search = []
                if transform_name == "simple" and "no_template" not in fcegan_name:
                    extra_search.append("simple_fcegan")
                elif transform_name == "ctgan" and "no_template" not in fcegan_name:
                    extra_search.append("/fcegan/fcegan")
                elif transform_name == "simple" and "no_template" in fcegan_name:
                    extra_search.append("simple_no_template")
                elif transform_name == "ctgan" and "no_template" in fcegan_name:
                    extra_search.append("/fcegan/no_template_fcegan")

                model_path = get_model(
                    output_dir / dataset_name,
                    *extra_search,
                    f"trial{trial}",
                    ".pth",
                    **search_params,
                )
                fcegan.load_state_dict(get_checkpoint(model_path))

                dataset_flexibilities[fcegan_name] = calculate_flexibility(
                    fcegan=fcegan,
                    data=test,
                    rounding_dict=rounding_dict,
                    switch_target_fn=switch_target_fn,
                    classifier=classifier,
                    eval_ctgan=eval_ctgan,
                    batch_size=batch_size,
                )

            print("random")
            dataset_flexibilities["random"] = calculate_flexibility_random_generator(
                fcegan=fcegan,
                data=test,
                rounding_dict=rounding_dict,
                switch_target_fn=switch_target_fn,
                classifier=classifier,
                eval_ctgan=eval_ctgan,
                batch_size=batch_size,
            )

            print("optimization")
            Load training CTGAN
            if transform_name == "simple":
                train_ctgan = SimpleCtganSynthesizer()
                train_ctgan.load_state_dict(
                    torch.load(
                        get_model(
                            output_dir / dataset_name,
                            f"{transform_name}_prediction_train_simple_ctgan",
                            ".pth",
                        )
                    )
                )
            else:
                train_ctgan = CtganSynthesizer()
                train_ctgan.load_state_dict(
                    torch.load(
                        get_model(
                            output_dir / dataset_name,
                            f"{transform_name}_prediction_train_ctgan",
                            ".pth",
                        )
                    )
                )

            for optimization_name, hyperparameters in tqdm(
                optimizations.items(), total=len(optimizations)
            ):
                dataset_flexibilities[optimization_name] = calculate_flexibility_optimizer(
                    data=test,
                    training_ctgan=train_ctgan,
                    eval_ctgan=eval_ctgan,
                    rounding_dict=rounding_dict,
                    fcegan=fcegan,
                    switch_target_fn=switch_target_fn,
                    classifier=classifier,
                    batch_size=batch_size,
                    # realism_influence=realism_infl,
                    **hyperparameters,
                )

            flexibilities[transform_name][dataset_name] = dataset_flexibilities

            # Save results
            print("Saving intermediary results")
            output_path = results_dir / f"flexibilities{trial}.pkl"
            if not output_path.exists():
                results_dir.mkdir(parents=True, exist_ok=True)
                with open(output_path, "wb") as file:
                    pickle.dump(flexibilities, file)
            else:
                # Only update the new results
                with open(output_path, "rb") as file:
                    old_flexibilities = pickle.load(file)
                for transform in flexibilities.keys():
                    for dataset in flexibilities[transform].keys():
                        for key, value in flexibilities[transform][dataset].items():
                            old_flexibilities[transform][dataset][key] = value

                with open(output_path, "wb") as file:
                    pickle.dump(old_flexibilities, file)
