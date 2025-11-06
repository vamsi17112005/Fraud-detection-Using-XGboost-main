import joblib

# Save training columns for future batch prediction alignment
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

DATA_PATH = '/Users/arinpatel/Downloads/Fraud.csv'
df = pd.read_csv(DATA_PATH)
drop_cols = ['nameOrig', 'nameDest']
df = df.drop(columns=drop_cols)
df = pd.get_dummies(df, columns=['type'], drop_first=True)
target = 'isFraud'
X = df.drop(columns=[target])
y = df[target]
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
smote = SMOTE(random_state=42, n_jobs=-1)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
joblib.dump(X_train_bal.columns.tolist(), 'xgboost_train_columns.joblib')
print('Saved training columns for batch prediction alignment.')
