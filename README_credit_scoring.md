# Credit Scoring Model (Synthetic, End-to-End)

This project builds a creditworthiness classifier using synthetic financial data.

## Contents
- `credit_scoring_dataset.csv`: dataset
- `models/LogisticRegression.joblib`, `models/DecisionTree.joblib`, `models/RandomForest.joblib`: trained models
- `roc_*.png`: ROC curves per model
- `feature_importances_random_forest.png`: global feature importances
- `credit_scoring_report.txt`: metrics and confusion matrices
- `run_inference.py`: quick example on how to load and score

## Quick Start
```bash
pip install -r requirements.txt  # sklearn, pandas, matplotlib, joblib, numpy
python run_inference.py
```
