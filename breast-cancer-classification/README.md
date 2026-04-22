# Diagnostic Breast Cancer Classification with PyTorch

A compact machine learning project that builds a feedforward neural network in PyTorch to classify breast cancer cases as **benign** or **malignant** using tabular diagnostic features.

## Overview

This notebook walks through a complete supervised classification workflow on the **Diagnostic Breast Cancer Dataset** from Kaggle. The focus is on a clean, readable, and reproducible binary classifier for tabular data, including:

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
- SHAP feature importance: global bar plot, beeswarm, per-sample waterfall, and feature-level scatter plots

The goal is not to build the most complex model possible, but to demonstrate a clear, reproducible, and interpretable end-to-end workflow in Python.

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

### 7. SHAP Analysis

Four types of SHAP visualisation are used to interrogate the model at different levels:

| Plot | Level | What it shows |
|------|-------|---------------|
| Summary bar | Global | Mean absolute SHAP value per feature, ranked by overall importance |
| Beeswarm | Global | SHAP value per sample per feature, coloured by feature value |
| Waterfall | Local (single sample) | How each feature pushed one prediction away from the baseline |
| Scatter | Feature-level | Relationship between a feature's actual value and its SHAP value across all samples |

`shap.DeepExplainer` is used to compute SHAP values. Waterfall plots are shown for one representative benign and one representative malignant sample, selected deterministically from the test set. Scatter plots are shown for three features that illustrate contrasting relationships: Mean Area (positive monotonic), Worst Symmetry (positive monotonic), and Fractal Dimension SE (negative monotonic, with a notable outlier).

## Results

The model correctly classified 112 out of 114 held-out samples, achieving 98% accuracy. Per-class performance:

| Class     | Precision | Recall | F1-score | Support |
|-----------|-----------|--------|----------|---------|
| Benign    | 0.99      | 0.99   | 0.99     | 72      |
| Malignant | 0.98      | 0.98   | 0.98     | 42      |

Only one malignant case in 42 was missed. In a diagnostic context, missed malignancies matter more than false alarms, so recall on the malignant class is the number to watch.

SHAP identified **Mean Area** as the feature with the largest influence on predictions, followed by Mean Texture, Mean Radius, and Worst Concave Points. This is consistent with how pathologists read FNA slides: larger nuclei with irregular shapes tend to indicate malignancy. Mean Area, Mean Radius, and Mean Perimeter all measure variations of the same thing and should be read together rather than separately.

The scatter plots showed that most features have clean monotonic relationships with the prediction. Mean Area and Worst Symmetry push toward malignant as their values increase; Fractal Dimension SE and Mean Fractal Dimension push toward benign. One outlier in Fractal Dimension SE sits well outside the training distribution and warrants attention if the model is applied to new data.

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
- add k-fold cross-validation for a more robust performance estimate
- tune the decision threshold to reduce missed malignancies further
- refactor the training loop to use `TensorDataset` and `DataLoader`

## Purpose

This project demonstrates:

- end-to-end ML workflow for tabular classification
- clean and reproducible notebook structure
- PyTorch fundamentals for binary classification
- correct train/test data handling and scaling
- evaluation with clinically relevant metrics
- model interpretability with SHAP across global, local, and feature-level views
