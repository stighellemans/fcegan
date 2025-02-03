# Flexible Counterfactual Explanations with Generative Models

This repository contains the implementation of **Flexible Counterfactual Explanations (FCEGAN)**, a framework designed to provide actionable insights by generating counterfactual explanations with user-defined mutable features at inference time. This approach is particularly useful in high-stakes domains like finance and healthcare, where interpretability and flexibility are crucial.

## 🔍 Overview

Counterfactual explanations suggest minimal changes to input features that alter a model’s prediction toward a desired outcome. Traditional methods use fixed sets of mutable features, limiting their adaptability. **FCEGAN** addresses this limitation by:

- 📌 **Introducing Flexible Counterfactual Templates** – Users can dynamically specify mutable and immutable features at inference time.
- 🏗 **Leveraging Generative Adversarial Networks (GANs)** – To generate realistic counterfactuals without requiring access to model internals.
- ⚫ **Supporting Black-Box Models** – Generating explanations using historical prediction datasets rather than direct access to the classification model.
- 🎯 **Ensuring High-Quality Counterfactuals** – Through a two-stage process that first generates candidates and then selects the most realistic and actionable ones.

## 🚀 Key Features

- **Flexible Counterfactual Explanations**: User-defined constraints for personalized counterfactual generation.
- **Black-Box Compatibility**: Generates explanations without direct access to the underlying model.
- **Two-Stage Selection Process**: Ensures counterfactuals meet predefined quality measures like realism, diversity, and actionability.
- **Generalizable Framework**: Applicable across various datasets and machine learning models.

## 🏗 Implementation

FCEGAN consists of the following core components:

1. **Data Preparation** (`preparation.ipynb`): Handles dataset preprocessing, including downloading datasets and formatting them for training.
2. **Training Counterfactual Models** (`train_experiments.py`): Implements the training of FCEGAN models using different classifiers and datasets.
3. **Generating Counterfactual Explanations** (`calculate_flexibilities.py`): Runs the counterfactual generation process with different model configurations. `flexibility.py` serves as the core  module to calculate flexibilities.

## 📦 Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook
- Dependencies listed in `requirements.txt`

### Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/stighellemans/fcegan.git
   cd fcegan
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

### Usage

The repository includes Jupyter Notebooks in `./examples` demonstrating how to use FCEGAN for counterfactual explanations.

1. **Prepare the Dataset**:
   Run `preparation.ipynb` to download and process the required datasets.

2. **Train the Model**:
   Use `train_experiments.py` to train FCEGAN on datasets such as the Adult UCI Income dataset or the Heart Disease Risk Prediction dataset.

3. **Generate Counterfactuals**:
   After training, run `calculate_flexibilities.py` to generate counterfactual explanations with different flexibility constraints.

## 📊 Datasets

FCEGAN has been tested on multiple datasets to evaluate its performance in generating realistic and actionable counterfactual explanations. These include:

- **Adult Census Income Dataset** (Kaggle: `uciml/adult-census-income`): Used to study income prediction and counterfactual explanations for socio-economic factors.
- **Heart Disease Risk Prediction Dataset** (Kaggle: `alphiree/cardiovascular-diseases-risk-prediction-dataset`): Evaluates counterfactual explanations in predicting cardiovascular disease risks.
- **Diabetes Health Indicators Dataset** (Kaggle: `alexteboul/diabetes-health-indicators-dataset`): Explores counterfactual explanations for diabetes risk assessment, incorporating categorical mappings for clearer interpretations.
- **Employee Attrition Dataset** (Kaggle: `stealthtechnologies/employee-attrition-dataset`): Used to assess job attrition predictions and generate counterfactual explanations for HR decision-making.

These datasets help validate FCEGAN’s effectiveness in multiple domains, ensuring robust and interpretable counterfactual reasoning across different prediction tasks.

## 📄 Paper

For an in-depth explanation of the methodology and experiments, refer to our [paper](./xxx.pdf). Additionally, explore `plotting.ipynb` for an overview of the figures used in the paper.

## 🏆 Results

FCEGAN has been extensively tested on multiple datasets, demonstrating superior performance in generating valid counterfactuals compared to existing methods.

## 🤝 Contributing

We welcome contributions to improve this framework. Feel free to fork the repository, submit issues, or create pull requests for feature enhancements and bug fixes.

## 📜 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

## 🙌 Acknowledgments

We appreciate the support and feedback from contributors and the research community in refining this framework.

