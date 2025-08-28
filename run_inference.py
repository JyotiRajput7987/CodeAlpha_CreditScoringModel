import joblib
import numpy as np

# Load the trained model
model = joblib.load("Credit_scoring_model.ipynb")

# Example input data [income, debts, payment_history_score]
# You can change values for demo
sample_input = np.array([[50000, 10000, 0.9]])  # 0.9 means 90% good payment history

# Predict
prediction = model.predict(sample_input)
prediction_proba = model.predict_proba(sample_input)

# Show output
print("Input Data: Income=50000, Debts=10000, Payment History=90%")
print(f"Predicted Creditworthiness: {'Approved' if prediction[0]==1 else 'Rejected'}")
print(f"Prediction Probability: {prediction_proba[0][1]*100:.2f}%")
