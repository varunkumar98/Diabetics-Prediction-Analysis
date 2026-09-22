# Diabetes Prediction Analysis

A data mining project comparing nine classification algorithms to predict diabetes onset from diagnostic health measurements, with hyperparameter tuning and feature-importance analysis on the best-performing model.

![Language](https://img.shields.io/badge/Language-Python-3776AB?logo=python&logoColor=white)
![ML](https://img.shields.io/badge/ML-scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-Pima%20Indians%20Diabetes-blue)

## Overview

This project applies data mining techniques to the **Pima Indians Diabetes dataset** (768 patient records, 8 diagnostic features) to predict whether a patient has diabetes (`Outcome`: 1) or not (`Outcome`: 0). It was completed as the final project for **IE7275: Data Mining in Engineering** at Northeastern University.

The dataset features are: `Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, and `Age`.

## Approach

1. **Data exploration** — 768 records × 9 columns, no `NaN` values, but zero-encoded missing values in `Glucose`, `BloodPressure`, `BMI`, and `Insulin`.
2. **Data visualization** — class balance (more non-diabetic than diabetic patients) and a correlation heatmap, which flagged `Glucose`, `BMI`, `Age`, and `Insulin` as the features most correlated with `Outcome`.
3. **Data processing** — handling of the zero-encoded missing values and `MinMaxScaler` normalization.
4. **Model exploration** — nine classifiers trained and compared on an 80/20 train/test split (614 / 154 samples):
   - Gaussian Naive Bayes
   - Bernoulli Naive Bayes
   - Logistic Regression
   - Random Forest
   - Support Vector Machine
   - Decision Tree
   - K-Nearest Neighbors
   - Gradient Boosting
   - Stochastic Gradient Descent
5. **Hyperparameter tuning** — k-fold cross-validation and grid search on the top-performing model.
6. **Performance diagnostics** — confusion matrices, ROC/AUC curves, and a feature-importance plot (via `SelectFromModel`) run across Random Forest, Decision Tree, and Gradient Boosting.

## Results

Random Forest and KNN were the top performers at **~75.97% accuracy** on the held-out test set; Random Forest was carried forward for k-fold cross-validation, ROC/AUC analysis, and feature-importance diagnostics. A separate untuned Random Forest run (with `Train score: 1.0`, indicating overfitting on the full-depth default tree) scored **75.32%** on test — underscoring why the project moves on to cross-validated hyperparameter tuning rather than trusting a single train/test split.

Feature-importance analysis consistently pointed to **Glucose, BMI, and Age** as the strongest predictors of diabetes outcome, consistent with the earlier correlation heatmap.

## Repository Contents

| File | Description |
|---|---|
| `Diabetes_Prediction_VK.ipynb` | Full analysis notebook: EDA, visualization, preprocessing, model comparison, tuning, and diagnostics. |
| `Varun_Final_Report_DataMining.pdf` | Final written report for IE7275. |
| `diabetes2.csv` | The Pima Indians Diabetes dataset (768 rows × 9 columns). |

## Tech Stack

- Python — `pandas`, `numpy`, `matplotlib`, `seaborn`
- `scikit-learn` — preprocessing, model training, `GridSearchCV`, `SelectFromModel`, evaluation metrics
- `pandas-profiling` — automated EDA report

## Getting Started

```bash
pip install pandas numpy matplotlib seaborn scikit-learn pandas-profiling jupyter
jupyter notebook Diabetes_Prediction_VK.ipynb
```

The notebook reads `diabetes2.csv` directly from the repository root, so no additional data download is needed.

## Author

Varun Kumar Kumaravel — IE7275: Data Mining in Engineering, Northeastern University (April 2023).

## License

No license is currently specified for this repository.
