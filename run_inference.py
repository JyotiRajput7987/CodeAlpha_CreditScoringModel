import pandas as pd
import joblib

"
            "df = pd.read_csv('credit_scoring_dataset.csv')
"
            "X = df.drop(columns=['creditworthy'])
"
            "rf = joblib.load('models/RandomForest.joblib')
"
            "pred = rf.predict(X.head(10))
"
            "proba = rf.predict_proba(X.head(10))[:,1]
"
            "print('Predictions:', pred)
"
            "print('Probabilities:', proba)
