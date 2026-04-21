# Diagnostic Breast Cancer Classification with PyTorch

A small machine learning project that builds a feedforward neural network in PyTorch to classify breast cancer cases as **benign** or **malignant** using tabular diagnostic features.

## Overview

This notebook walks through a complete supervised classification workflow on the **Diagnostic Breast Cancer Dataset** from Kaggle. The project focuses on a clean, readable implementation of a binary classifier for tabular data, including:

- loading the dataset from Kaggle
- basic data inspection
- label encoding
- feature selection
- train/test split with stratification
- feature normalisation
- building a simple PyTorch neural network
- model training and evaluation
- visualisation of training curves
- confusion matrix and classification report

The goal here is not to build the most complex model possible, but to demonstrate a clear and reproducible end-to-end ML workflow in Python.

## Dataset

**Source:** Kaggle  
**Dataset:** Diagnostic Breast Cancer Dataset

The dataset contains diagnostic measurements derived from breast cancer samples and a target label indicating whether each case is:

- **Benign**
- **Malignant**

In this notebook:

- `Diagnosis` is mapped to binary labels
- `ID` and `Diagnosis` are excluded from the feature matrix
- the remaining numeric columns are used as model inputs

## Project Structure

This project is currently contained in a single Jupyter notebook:

- `Diagnostic Breast Cancer Dataset.ipynb`

## Methods

### 1. Data preparation

The notebook:

- downloads the dataset with `kagglehub`
- reads the CSV file into a pandas DataFrame
- checks shape, missing values, summary statistics, and column names
- maps labels:
  - `Benign -> 0`
  - `Malignant -> 1`

### 2. Train/test split and scaling

The data is split into training and test sets using `train_test_split`, with stratification applied to preserve class balance.

Features are then standardised using `StandardScaler`.

### 3. Model

The classifier is a simple feedforward neural network built with `torch.nn.Sequential`, using:

- input layer matching the number of features
- hidden layer with ReLU activation
- output layer for binary classification

### 4. Training

The model is trained with:

- **Loss:** CrossEntropyLoss
- **Optimiser:** Adam
- **Epochs:** 100

Training and test loss/accuracy are tracked across epochs.

### 5. Evaluation

Model evaluation includes:

- training and test loss curves
- training and test accuracy curves
- confusion matrix
- classification report

## Results

The model achieved strong performance on the held-out test set, reaching approximately **98% test accuracy** by the end of training.

The notebook also includes:

- a confusion matrix for class-wise performance
- a classification report summarising precision, recall, and F1-score

These results suggest that even a relatively small neural network can perform very well on this dataset.

## Requirements

Install the main dependencies before running the notebook:

```bash
pip install torch pandas numpy matplotlib scikit-learn kagglehub
```

## How to Run

1. Clone this repository
2. Open the notebook in JupyterLab or Jupyter Notebook
3. Run the cells from top to bottom

The dataset is downloaded inside the notebook using `kagglehub`, so there is no need to store the CSV manually in the repository unless you prefer to.

## Possible Next Steps

Some natural extensions for this project would be:

- compare PyTorch against simpler baselines such as logistic regression and random forest
- add cross-validation for more robust evaluation
- tune model architecture and hyperparameters
- package the workflow into a reusable Python script or small app
- add SHAP or feature importance analysis for interpretability

## Purpose

This is a compact portfolio project designed to demonstrate:

- practical ML workflow design
- clean notebook structure
- PyTorch fundamentals for tabular classification
- sensible evaluation and visualisation
- reproducible analysis in Python

