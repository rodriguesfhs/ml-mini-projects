# Diagnostic Breast Cancer Classification with PyTorch

A compact machine learning project that builds a feedforward neural network in PyTorch to classify breast cancer cases as **benign** or **malignant** using tabular diagnostic features.

## Overview

This notebook walks through a complete supervised classification workflow on the **Diagnostic Breast Cancer Dataset** from Kaggle. The project focuses on a clean, readable, and reproducible implementation of a binary classifier for tabular data, including:

- loading the dataset from Kaggle
- basic data inspection
- label encoding
- feature selection
- reproducible seeding
- train/test split with stratification
- feature normalisation (fit on train only)
- building a simple PyTorch neural network
- model training and evaluation
- visualisation of training curves
- confusion matrix and classification report
- SHAP-based feature importance analysis

The goal is not to build the most complex model possible, but to demonstrate a clear, reproducible, and interpretable end-to-end ML workflow in Python.

## Dataset

**Source:** Kaggle  
**Dataset:** [Diagnostic Breast Cancer Dataset](https://www.kaggle.com/datasets/ahmeduzaki/diagnostic-breast-cancer-dataset)

The dataset contains 569 samples and 30 numerical features describing characteristics of cell nuclei extracted from fine needle aspirate (FNA) biopsy images, such as radius, texture, perimeter, area, and smoothness; each recorded as mean, standard error, and worst value.

Target label:
- **Benign**
- **Malignant**

In this notebook:
- `Diagnosis` is mapped to binary labels (`Benign → 0`, `Malignant → 1`)
- `ID` and `Diagnosis` are excluded from the feature matrix
- the remaining 30 numeric columns are used as model inputs

## Project Structure

```
breast-cancer-classification/
└── Diagnostic Breast Cancer Dataset.ipynb
```

## Methods

### 1. Reproducibility

A global seed (`SEED = 42`) is set across `random`, `numpy`, and `torch` before any data splitting or model initialisation.

### 2. Data preparation

- dataset downloaded via `kagglehub`
- shape, missing values, summary statistics, and column names inspected
- labels encoded: `Benign → 0`, `Malignant → 1`
- `ID` and `Diagnosis` columns dropped

### 3. Train/test split and scaling

Data is split 80/20 with stratification to preserve class balance. Features are standardised using `StandardScaler`, fitted on the training set only and applied to both splits.

### 4. Model

A simple feedforward neural network built with `torch.nn.Module`:

```
Input (30) → Linear → ReLU → Linear → Output (2)
             [64 units]
```

### 5. Training

- **Loss:** `CrossEntropyLoss`
- **Optimiser:** Adam (`lr=1e-3`)
- **Epochs:** 100
- Training and test loss/accuracy tracked across all epochs

### 6. Evaluation

- training and test loss curves
- training and test accuracy curves
- confusion matrix
- classification report (precision, recall, F1-score per class)

### 7. SHAP Feature Importance

`shap.DeepExplainer` is used to compute SHAP values for the malignant class across the test set. Results are visualised as a global bar chart of mean absolute SHAP values, ranking all 30 features by their average contribution to the model's predictions.

## Results

The model achieved **98% accuracy** on the held-out test set (114 samples), with the following per-class performance:

| Class     | Precision | Recall | F1-score | Support |
|-----------|-----------|--------|----------|---------|
| Benign    | 0.99      | 0.99   | 0.99     | 72      |
| Malignant | 0.98      | 0.98   | 0.98     | 42      |

In a clinical screening context, recall on the malignant class is the critical metric, as false negatives carry greater consequence than false positives. Only one malignant case in 42 was missed.

SHAP analysis identified **Mean Area** as the strongest predictor of malignancy, followed by Mean Texture, Worst Smoothness, Mean Radius, and Worst Concave Points; consistent with the known morphological basis of FNA-based diagnosis. Note that several top features (Mean Area, Mean Radius, Mean Perimeter) are correlated size descriptors and should be interpreted collectively.

## Requirements

```bash
pip install torch pandas numpy matplotlib scikit-learn kagglehub shap
```

## How to Run

1. Clone this repository
2. Open the notebook in JupyterLab or Jupyter Notebook
3. Run the cells from top to bottom

The dataset is downloaded automatically inside the notebook via `kagglehub`. No manual CSV download required.

## Potential Next Steps

- compare against logistic regression and random forest baselines
- add k-fold cross-validation for more robust performance estimates
- tune the decision threshold to maximise malignant recall
- refactor training loop to use `TensorDataset` and `DataLoader`

## Purpose

This is a compact portfolio project designed to demonstrate:

- practical ML workflow design
- clean and reproducible notebook structure
- PyTorch fundamentals for tabular classification
- correct train/test data handling and scaling
- sensible evaluation with clinically relevant metrics
- model interpretability with SHAP
