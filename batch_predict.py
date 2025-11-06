import pandas as pd
import joblib
from xgboost import XGBClassifier

# Load model and new data
MODEL_PATH = 'xgboost_fraud_model.joblib'
model = joblib.load(MODEL_PATH)

# Example: load new data (replace with your new data path)
NEW_DATA_PATH = '/Users/arinpatel/Downloads/New_Fraud_Data.csv'  # Update as needed
new_df = pd.read_csv(NEW_DATA_PATH)

# Preprocess new data (drop names, encode type)
drop_cols = ['nameOrig', 'nameDest']
for col in drop_cols:
    if col in new_df.columns:
        new_df = new_df.drop(columns=[col])
if 'type' in new_df.columns:
    new_df = pd.get_dummies(new_df, columns=['type'], drop_first=True)

# Align columns with training data
import numpy as np
import os
X_train_cols = joblib.load('xgboost_train_columns.joblib') if os.path.exists('xgboost_train_columns.joblib') else None
if X_train_cols is not None:
    for col in X_train_cols:
        if col not in new_df.columns:
            new_df[col] = 0
    new_df = new_df[X_train_cols]

# Predict
preds = model.predict(new_df)
probs = model.predict_proba(new_df)[:,1]

# Save predictions
output = new_df.copy()
output['isFraud_pred'] = preds
output['fraud_prob'] = probs
output.to_csv('fraud_predictions.csv', index=False)
print('Predictions saved to fraud_predictions.csv')
