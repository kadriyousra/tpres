import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
import time

# Load dataset
data = load_breast_cancer()
X = data.data  # Features
y = data.target  # Labels (0 = malignant, 1 = benign)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Function to evaluate models
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"Accuracy: {accuracy:.2f}")
    print(f"ROC AUC: {roc_auc:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    return y_proba

# GridSearchCV for Random Forest
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10]
}
rf_model = RandomForestClassifier(random_state=42)
grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid_rf, scoring='accuracy', cv=5, verbose=1, n_jobs=-1)
start_time = time.time()
grid_search_rf.fit(X_train, y_train)
runtime_rf = time.time() - start_time
best_rf = grid_search_rf.best_estimator_
y_proba_rf = evaluate_model(best_rf, X_test, y_test)
print(f"Best Parameters for RF: {grid_search_rf.best_params_}")
print(f"Training Time: {runtime_rf:.2f} seconds\n")

# GridSearchCV for XGBoost
param_grid_xgb = {
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'n_estimators': [50, 100, 200]
}
xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
grid_search_xgb = GridSearchCV(estimator=xgb_model, param_grid=param_grid_xgb, scoring='accuracy', cv=5, verbose=1, n_jobs=-1)
start_time = time.time()
grid_search_xgb.fit(X_train, y_train)
runtime_xgb = time.time() - start_time
best_xgb = grid_search_xgb.best_estimator_
y_proba_xgb = evaluate_model(best_xgb, X_test, y_test)
print(f"Best Parameters for XGB: {grid_search_xgb.best_params_}")
print(f"Training Time: {runtime_xgb:.2f} seconds\n")

# Compare Feature Importance
rf_importances = best_rf.feature_importances_
xgb_importances = best_xgb.feature_importances_
rf_indices = np.argsort(rf_importances)[::-1]
xgb_indices = np.argsort(xgb_importances)[::-1]

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Feature Importances (Random Forest)")
plt.bar(range(X.shape[1]), rf_importances[rf_indices], align="center")
plt.xticks(range(X.shape[1]), [data.feature_names[i] for i in rf_indices], rotation=90)
plt.xlabel("Feature")
plt.ylabel("Importance")

plt.subplot(1, 2, 2)
plt.title("Feature Importances (XGBoost)")
plt.bar(range(X.shape[1]), xgb_importances[xgb_indices], align="center", color='orange')
plt.xticks(range(X.shape[1]), [data.feature_names[i] for i in xgb_indices], rotation=90)
plt.xlabel("Feature")
plt.ylabel("Importance")

plt.tight_layout()
plt.show()

# Plot ROC Curve
plt.figure(figsize=(8, 6))
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_proba_xgb)
plt.plot(fpr_rf, tpr_rf, label="Random Forest (AUC = {:.2f})".format(roc_auc_score(y_test, y_proba_rf)))
plt.plot(fpr_xgb, tpr_xgb, label="XGBoost (AUC = {:.2f})".format(roc_auc_score(y_test, y_proba_xgb)))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
