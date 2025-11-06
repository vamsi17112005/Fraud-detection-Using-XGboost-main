from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

# Random Forest
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_probs = rf.predict_proba(X_test)[:,1]
print('\nRandom Forest Results:')
print('Accuracy:', accuracy_score(y_test, rf_preds))
print('ROC AUC:', roc_auc_score(y_test, rf_probs))
print(classification_report(y_test, rf_preds))

# LightGBM
lgbm = LGBMClassifier(n_estimators=100, n_jobs=-1, random_state=42)
lgbm.fit(X_train, y_train)
lgbm_preds = lgbm.predict(X_test)
lgbm_probs = lgbm.predict_proba(X_test)[:,1]
print('\nLightGBM Results:')
print('Accuracy:', accuracy_score(y_test, lgbm_preds))
print('ROC AUC:', roc_auc_score(y_test, lgbm_probs))
print(classification_report(y_test, lgbm_preds))
