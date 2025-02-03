from typing import Callable, Dict

import numpy as np
import pandas as pd
import torch

from counterfactual_package.data.utils import (
    corrupt_dataframe_with_nan,
    transform_mask,
)
from counterfactual_package.engines.counterfactual_synthesizer import (
    FCEGAN,
    NoTemplateFCEGAN,
    evaluate_counterfactuals,
)
from counterfactual_package.engines.ctgan_synthesizer import (
    CtganSynthesizer,
)
from counterfactual_package.engines.optimization import (
    CounterfactualOptimizer,
)
from counterfactual_package.engines.training import (
    calc_fcegan_diversity,
    calc_fcegan_diversity_random_generator,
    calc_optimization_diversity,
)
from counterfactual_package.models.classifiers import Classifier
from counterfactual_package.utils.utils import (
    check_available_device,
)


def calculate_flexibility(
    fcegan: FCEGAN,
    data: pd.DataFrame,
    rounding_dict: Dict[str, int],
    switch_target_fn: Callable[[str], str],
    classifier: Classifier,
    eval_ctgan: CtganSynthesizer,
    batch_size=300,
):
    classifier.to("cpu")
    target_name = fcegan.config["target_name"]

    flexibility = []
    for corruption in np.arange(0.0, 1.1, 0.1):

        samples = data.copy().reset_index(drop=True)
        cf_template = corrupt_dataframe_with_nan(
            samples, corruption_percentage=corruption, exclude=[target_name]
        )
        cf_template[target_name] = cf_template[target_name].apply(switch_target_fn)

        to_change_mask = transform_mask(
            torch.tensor(cf_template.isna().to_numpy(), dtype=torch.bool),
            fcegan.metadata,
        )

        samples_tf = fcegan.transform(samples)

        cf_templates_tensor = samples_tf.masked_fill(to_change_mask, 0)

        # change to right counterfactual target
        target_idxs = fcegan.metadata.column_to_transformed_idxs(target_name)
        cf_targets = fcegan.transform(cf_template[target_name])
        cf_templates_tensor[:, target_idxs] = cf_targets

        if isinstance(fcegan, (NoTemplateFCEGAN)):
            raw_counterfactuals_tf = fcegan.model.predict(samples_tf, samples_tf)
        else:
            raw_counterfactuals_tf = fcegan.model.predict(
                samples_tf, cf_templates_tensor
            )

        raw_counterfactuals = fcegan.reverse_transform(raw_counterfactuals_tf)
        counterfactuals = fcegan.post_process_counterfactuals(
            raw_counterfactuals, cf_template
        )

        for cont_col, rounding in rounding_dict.items():
            counterfactuals[cont_col] = counterfactuals[cont_col].round(rounding)

        all_metrics = fcegan.evaluate_counterfactuals(
            counterfactuals, samples, classifier, eval_ctgan, batch_size=batch_size
        )

        metrics = all_metrics["global_metrics"]

        # Select only the counterfactuals
        # If no valid counterfactuals, set valid_fraction to 0
        # And calculate the metrics with the invalid metrics!
        if metrics["valid_counterfactuals"].sum() == 0:
            metrics["valid_counterfactuals"] = True
            valid_metrics = metrics
            valid_metrics["valid_fraction"] = 0
        else:
            valid_metrics = metrics[
                metrics["valid_counterfactuals"] == 1.0
            ].reset_index(drop=True)
            valid_metrics["valid_fraction"] = metrics[
                "valid_counterfactuals"
            ].sum() / len(counterfactuals)

        # Find valid counterfactual template frequencies
        valid_counterfactual_template_frequencies = (
            cf_template[metrics["valid_counterfactuals"] == 1.0].isna().mean(axis=0)
        )
        valid_counterfactual_template_frequencies.name = corruption
        valid_counterfactual_template_frequencies["corruption"] = corruption

        # Calculate diversity for a couple of random samples
        diversity_nums = 10
        diversity_metrics = []
        for _ in range(diversity_nums):
            i = torch.randint(0, len(samples), (1,)).item()
            diversity_metrics.append(
                calc_fcegan_diversity(
                    fcegan, samples.iloc[i], batch_size=30, dropout=corruption
                )
            )
        mean_diversity = pd.DataFrame(diversity_metrics).mean()
        valid_metrics["categorical_diversity"] = mean_diversity["categorical_diversity"]
        valid_metrics["continuous_diversity"] = mean_diversity["continuous_diversity"]

        valid_metrics["corruption"] = corruption

        valid_metrics = valid_metrics.groupby(["corruption"]).agg(
            cat_changed=("cat_changed", "mean"),
            cat_changed_sem=("cat_changed", "sem"),
            mean_percentile_shift=("mean_percentile_shift", "mean"),
            mean_percentile_shift_sem=("mean_percentile_shift", "sem"),
            max_percentile_shift=("max_percentile_shift", "mean"),
            max_percentile_shift_sem=("max_percentile_shift", "sem"),
            counterfactual_prediction=("counterfactual_pred", "mean"),
            counterfactual_prediction_sem=("counterfactual_pred", "sem"),
            prediction_gain=("prediction_gain", "mean"),
            prediction_gain_sem=("prediction_gain", "sem"),
            fakeness=("fakeness", "mean"),
            fakeness_sem=("fakeness", "sem"),
            categorical_diversity=("categorical_diversity", "mean"),
            categorical_diversity_sem=("categorical_diversity", "sem"),
            continuous_diversity=("continuous_diversity", "mean"),
            continuous_diversity_sem=("continuous_diversity", "sem"),
            valid_fraction=("valid_fraction", "mean"),
            valid_fraction_sem=("valid_fraction", "sem"),
        )

        flexibility.append(
            pd.concat(
                [valid_metrics, valid_counterfactual_template_frequencies.to_frame().T],
                axis=1,
            )
        )

    return pd.concat(flexibility, axis=0)


def calculate_flexibility_random_generator(
    fcegan: FCEGAN,
    data: pd.DataFrame,
    rounding_dict: Dict[str, int],
    switch_target_fn: Callable[[str], str],
    classifier: Classifier,
    eval_ctgan: CtganSynthesizer,
    batch_size=300,
):
    target_name = fcegan.config["target_name"]

    classifier.to("cpu")
    flexibility = []
    for corruption in np.arange(0.0, 1.1, 0.1):

        samples = data.copy().reset_index(drop=True)
        cf_template = corrupt_dataframe_with_nan(
            samples, corruption_percentage=corruption, exclude=[target_name]
        )
        cf_template[target_name] = cf_template[target_name].apply(switch_target_fn)

        to_change_mask = transform_mask(
            torch.tensor(cf_template.isna().to_numpy(), dtype=torch.bool),
            fcegan.metadata,
        )

        samples_tf = fcegan.transform(samples)

        cf_templates_tensor = samples_tf.masked_fill(to_change_mask, 0)

        # change to right counterfactual target
        target_idxs = fcegan.metadata.column_to_transformed_idxs(target_name)
        cf_targets = fcegan.transform(cf_template[target_name])
        cf_templates_tensor[:, target_idxs] = cf_targets

        random_tensor = torch.rand_like(samples_tf)
        raw_counterfactuals = fcegan.reverse_transform(random_tensor)
        counterfactuals = fcegan.post_process_counterfactuals(
            raw_counterfactuals, cf_template
        )

        for cont_col, rounding in rounding_dict.items():
            counterfactuals[cont_col] = counterfactuals[cont_col].round(rounding)

        all_metrics = fcegan.evaluate_counterfactuals(
            counterfactuals, samples, classifier, eval_ctgan, batch_size=batch_size
        )

        metrics = all_metrics["global_metrics"]

        # Select only the counterfactuals
        # If no valid counterfactuals, set valid_fraction to 0
        # And calculate the metrics with the invalid metrics!
        if metrics["valid_counterfactuals"].sum() == 0:
            metrics["valid_counterfactuals"] = True
            valid_metrics = metrics
            valid_metrics["valid_fraction"] = 0
        else:
            valid_metrics = metrics[
                metrics["valid_counterfactuals"] == 1.0
            ].reset_index(drop=True)
            valid_metrics["valid_fraction"] = metrics[
                "valid_counterfactuals"
            ].sum() / len(counterfactuals)

        # Find valid counterfactual template frequencies
        valid_counterfactual_template_frequencies = (
            cf_template[metrics["valid_counterfactuals"] == 1.0].isna().mean(axis=0)
        )
        valid_counterfactual_template_frequencies.name = corruption
        valid_counterfactual_template_frequencies["corruption"] = corruption

        # Calculate diversity for a couple of random samples
        diversity_nums = 10
        diversity_metrics = []
        for _ in range(diversity_nums):
            i = torch.randint(0, len(samples), (1,)).item()
            diversity_metrics.append(
                calc_fcegan_diversity_random_generator(
                    fcegan, samples.iloc[i], batch_size=30, dropout=corruption
                )
            )
        mean_diversity = pd.DataFrame(diversity_metrics).mean()
        valid_metrics["categorical_diversity"] = mean_diversity["categorical_diversity"]
        valid_metrics["continuous_diversity"] = mean_diversity["continuous_diversity"]

        valid_metrics["corruption"] = corruption

        valid_metrics = valid_metrics.groupby(["corruption"]).agg(
            cat_changed=("cat_changed", "mean"),
            cat_changed_sem=("cat_changed", "sem"),
            mean_percentile_shift=("mean_percentile_shift", "mean"),
            mean_percentile_shift_sem=("mean_percentile_shift", "sem"),
            max_percentile_shift=("max_percentile_shift", "mean"),
            max_percentile_shift_sem=("max_percentile_shift", "sem"),
            counterfactual_prediction=("counterfactual_pred", "mean"),
            counterfactual_prediction_sem=("counterfactual_pred", "sem"),
            prediction_gain=("prediction_gain", "mean"),
            prediction_gain_sem=("prediction_gain", "sem"),
            fakeness=("fakeness", "mean"),
            fakeness_sem=("fakeness", "sem"),
            categorical_diversity=("categorical_diversity", "mean"),
            categorical_diversity_sem=("categorical_diversity", "sem"),
            continuous_diversity=("continuous_diversity", "mean"),
            continuous_diversity_sem=("continuous_diversity", "sem"),
            valid_fraction=("valid_fraction", "mean"),
            valid_fraction_sem=("valid_fraction", "sem"),
        )

        flexibility.append(
            pd.concat(
                [valid_metrics, valid_counterfactual_template_frequencies.to_frame().T],
                axis=1,
            )
        )

    return pd.concat(flexibility, axis=0)


def calculate_flexibility_optimizer(
    data: pd.DataFrame,
    training_ctgan: CtganSynthesizer,
    eval_ctgan: CtganSynthesizer,
    rounding_dict: Dict[str, int],
    fcegan: FCEGAN,
    switch_target_fn: Callable[[str], str],
    classifier: Classifier,
    template_guided: bool = True,
    classifier_influence=1,
    divergence_influence=10,
    realism_influence=0.1,
    betas=(0.9, 0.999),
    lr=1e-1,
    num_steps=20,
    batch_size=300,
):
    samples = data.copy().reset_index(drop=True)

    device = check_available_device()
    target_name = fcegan.config["target_name"]

    flexibility = []
    for corruption in np.arange(0.0, 1.1, 0.1):
        cf_template = corrupt_dataframe_with_nan(
            samples, corruption_percentage=corruption, exclude=[target_name]
        ).reset_index(drop=True)
        cf_template[target_name] = cf_template[target_name].apply(switch_target_fn)

        cf_optimizer = CounterfactualOptimizer(
            classifier=classifier,
            train_ctgan=training_ctgan,
            transform_fn=fcegan.transform,
            reverse_transform_fn=fcegan.reverse_transform,
            metadata=fcegan.metadata,
            lr=lr,
            betas=betas,
            classifier_influence=classifier_influence,
            divergence_influence=divergence_influence,
            realism_influence=realism_influence,
        )
        cf_optimizer.to(device)
        counterfactuals = cf_optimizer.optimize_counterfactuals(
            samples=samples,
            cf_templates=cf_template,
            target_name=target_name,
            num_steps=num_steps,
            template_guided=template_guided,
            evaluate_each=0,
            verbose=False,
        )
        for cont_col, rounding in rounding_dict.items():
            counterfactuals[cont_col] = counterfactuals[cont_col].round(rounding)

        all_metrics = evaluate_counterfactuals(
            counterfactuals=counterfactuals,
            originals=samples,
            classifier=classifier,
            classifier_transform=fcegan.transform,
            ctgan=eval_ctgan,
            metadata=fcegan.metadata,
            target_name=target_name,
            batch_size=batch_size,
        )

        metrics = all_metrics["global_metrics"]

        # Select only the counterfactuals
        # If no valid counterfactuals, set valid_fraction to 0
        # And calculate the metrics with the invalid metrics!
        if metrics["valid_counterfactuals"].sum() == 0:
            metrics["valid_counterfactuals"] = True
            valid_metrics = metrics
            valid_metrics["valid_fraction"] = 0
        else:
            valid_metrics = metrics[
                metrics["valid_counterfactuals"] == 1.0
            ].reset_index(drop=True)
            valid_metrics["valid_fraction"] = metrics[
                "valid_counterfactuals"
            ].sum() / len(counterfactuals)

        # Find valid counterfactual template frequencies
        valid_counterfactual_template_frequencies = (
            cf_template[metrics["valid_counterfactuals"] == 1.0].isna().mean(axis=0)
        )
        valid_counterfactual_template_frequencies.name = corruption
        valid_counterfactual_template_frequencies["corruption"] = corruption

        # Calculate diversity for a couple of random samples
        diversity_nums = 10
        diversity_metrics = []
        for _ in range(diversity_nums):
            i = torch.randint(0, len(samples), (1,)).item()
            diversity_metrics.append(
                calc_optimization_diversity(
                    cf_optimizer,
                    samples.iloc[i],
                    num_steps=num_steps,
                    batch_size=30,
                    dropout=corruption,
                )
            )
        mean_diversity = pd.DataFrame(diversity_metrics).mean()
        valid_metrics["categorical_diversity"] = mean_diversity["categorical_diversity"]
        valid_metrics["continuous_diversity"] = mean_diversity["continuous_diversity"]

        valid_metrics["corruption"] = corruption

        valid_metrics = valid_metrics.groupby(["corruption"]).agg(
            cat_changed=("cat_changed", "mean"),
            cat_changed_sem=("cat_changed", "sem"),
            mean_percentile_shift=("mean_percentile_shift", "mean"),
            mean_percentile_shift_sem=("mean_percentile_shift", "sem"),
            max_percentile_shift=("max_percentile_shift", "mean"),
            max_percentile_shift_sem=("max_percentile_shift", "sem"),
            counterfactual_prediction=("counterfactual_pred", "mean"),
            counterfactual_prediction_sem=("counterfactual_pred", "sem"),
            prediction_gain=("prediction_gain", "mean"),
            prediction_gain_sem=("prediction_gain", "sem"),
            fakeness=("fakeness", "mean"),
            fakeness_sem=("fakeness", "sem"),
            categorical_diversity=("categorical_diversity", "mean"),
            categorical_diversity_sem=("categorical_diversity", "sem"),
            continuous_diversity=("continuous_diversity", "mean"),
            continuous_diversity_sem=("continuous_diversity", "sem"),
            valid_fraction=("valid_fraction", "mean"),
            valid_fraction_sem=("valid_fraction", "sem"),
        )

        flexibility.append(
            pd.concat(
                [valid_metrics, valid_counterfactual_template_frequencies.to_frame().T],
                axis=1,
            )
        )

    return pd.concat(flexibility, axis=0)
