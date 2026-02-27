Random Forest End-to-End ML Pipeline using DVC
📌 Project Overview

This project implements a fully reproducible Machine Learning pipeline using:

Random Forest Classifier

DVC (Data Version Control)

Git for version control

The objective is to build an industry-style ML workflow with experiment tracking, parameter management, and reproducibility.

Dataset used:
Breast Cancer Wisconsin dataset from sklearn.

`PROJECT STRUCTURE`

random_forest_dvc/
│
├── data/
│   ├── raw/
│   │   └── data.csv
│   └── processed/
│       ├── train.csv
│       └── test.csv
│
├── src/
│   ├── data_load.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   ├── feature_importance.py
│   └── error_analysis.py
│
├── params.yaml
├── dvc.yaml
├── dvc.lock
├── metrics.json
├── model.pkl
├── feature_importance.csv
├── confusion_matrix.png
├── requirements.txt
└── README.md

Pipeline Stages
1️⃣ Load Data

Loads raw dataset
Performs basic validation
Saves cleaned dataset

2️⃣ Preprocess

Train/Test split (80/20)
No scaling (Random Forest does not require it)

3️⃣ Train

Trains RandomForestClassifier
Hyperparameters read from params.yaml
Saves model as model.pkl

4️⃣ Evaluate

Calculates:
Accuracy
Precision
Recall
F1-score
ROC-AUC
Saves metrics to metrics.json

5️⃣ Feature Importance

Extracts top 10 important features
Saves to feature_importance.csv

6️⃣ Error Analysis

Generates confusion matrix
Saves visualization as confusion_matrix.png

`Reproducing the Pipeline`

To run the full pipeline:

'dvc repro'

DVC automatically:

Detects changes
Re-runs only necessary stages
Ensures reproducibility


`Experiment Tracking`

Hyperparameters are defined in:

params.yaml

Example:

train:
  n_estimators: 200
  max_depth: 10
  min_samples_split: 2
  random_state: 42

When parameters are modified:

dvc repro

To compare experiments:

dvc metrics show
dvc metrics diff

`Model Performance`

Example results:
Accuracy: 0.960
ROC-AUC: 0.99
Strong recall for malignant class0
Very low false negatives0000
The model performs well in detecting cancerous tumors, which is critical in medical diagnosis.

`Key Learnings`

Building modular ML pipelines
Data versioning with DVC
Parameter tracking
Reproducibility in ML workflows
Feature importance interpretation
Confusion matrix based error analysis
Experiment comparison

`Technologies Used`

Python
scikit-learn
pandas
matplotlib
DVC
Git
