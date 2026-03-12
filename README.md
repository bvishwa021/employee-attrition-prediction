# employee-attrition-prediction

## Overview
An end-to-end machine learning project that predicts whether an employee is likely to leave a company (attrition) using the IBM HR Analytics dataset from Kaggle.
This is a binary classification problem — the target variable is `Attrition` (Yes/No).

## Project Structure
```
employee-attrition-prediction/
├── data/
│   ├── raw/              # Original dataset (never modified)
│   └── processed/        # Cleaned and encoded data
├── notebooks/            # Jupyter notebooks for EDA and modeling
├── src/                  # Reusable scripts and helper functions
├── outputs/
│   ├── figures/          # Saved plots and visualizations
│   ├── models/           # Saved trained models
│   └── reports/          # Evaluation metrics and reports
├── requirements.txt
└── README.md
```

## Pipeline
1. **EDA** — Understanding who leaves, income analysis, overtime patterns, correlations
2. **Preprocessing** — Encoding, scaling, train-test split, SMOTE for class imbalance
3. **Modeling** — Logistic Regression → Random Forest → XGBoost
4. **Evaluation** — ROC-AUC, F1-score, Confusion Matrix, Feature Importance

## Tech Stack
- Python
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- XGBoost
- imbalanced-learn (SMOTE)

## Dataset
[IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) — via Kaggle
