import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, f1_score
from imblearn.over_sampling import SMOTE
import joblib


# Config
DATA_PATH = '/content/drive/MyDrive/Data_sets/Fraud.csv'
MODEL_PATH = 'xgboost_fraud_model.joblib'
RANDOM_STATE = 42

# Load dataset
df = pd.read_csv(DATA_PATH)

# Display basic info
def basic_eda(df):
    print('Shape:', df.shape)
    print('Columns:', df.columns.tolist())
    print(df.head())
    print(df.describe())
    print(df.isnull().sum())


# Run EDA
basic_eda(df)

# Visualize class distribution
plt.figure(figsize=(6,4))
sns.countplot(x='isFraud', data=df)
plt.title('Class Distribution (isFraud)')
plt.show()

# Visualize transaction types
plt.figure(figsize=(8,4))
sns.countplot(x='type', data=df)
plt.title('Transaction Type Distribution')
plt.show()


# Preprocessing
# Drop columns that are not useful for modeling (IDs, names)
drop_cols = ['nameOrig', 'nameDest']
df = df.drop(columns=drop_cols)


# Remove outliers using IQR method for numerical columns
num_cols = df.select_dtypes(include=[np.number]).columns.difference(['isFraud'])
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower) & (df[col] <= upper)]

# Encode categorical 'type'
df = pd.get_dummies(df, columns=['type'], drop_first=True)


# Features and target
target = 'isFraud'
X = df.drop(columns=[target])
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

# Balance the training set with SMOTE
smote = SMOTE(random_state=RANDOM_STATE)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
print(f"After SMOTE: {sum(y_train_bal==1)} frauds, {sum(y_train_bal==0)} non-frauds")




# Train XGBoost model with class weighting
scale_pos_weight = sum(y_train_bal==0) / sum(y_train_bal==1)
xgb = XGBClassifier(tree_method='hist', use_label_encoder=False, eval_metric='logloss', n_jobs=-1, random_state=RANDOM_STATE, scale_pos_weight=scale_pos_weight)
xgb.fit(X_train_bal, y_train_bal)

# Cross-validation (StratifiedKFold)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scores = cross_val_score(xgb, X_train_bal, y_train_bal, cv=cv, scoring='f1')
print(f"5-Fold CV F1 scores: {cv_scores}")
print(f"Mean F1: {np.mean(cv_scores):.4f}")

# Predictions and evaluation
preds = xgb.predict(X_test)
probs = xgb.predict_proba(X_test)[:,1]
print('\nXGBoost Results:')
print('Accuracy:', accuracy_score(y_test, preds))
print('ROC AUC:', roc_auc_score(y_test, probs))
print(classification_report(y_test, preds))

# Save the XGBoost model
joblib.dump(xgb, MODEL_PATH)
print(f'\nXGBoost model saved as {MODEL_PATH}')