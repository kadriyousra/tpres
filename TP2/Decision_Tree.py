import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
import time
from sklearn.tree import DecisionTreeClassifier

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

# GridSearchCV for Decision Tree
param_grid_dt = {
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
dt_model = DecisionTreeClassifier(random_state=42)
grid_search_dt = GridSearchCV(estimator=dt_model, param_grid=param_grid_dt, scoring='accuracy', cv=5, verbose=1, n_jobs=-1)

start_time = time.time()
grid_search_dt.fit(X_train, y_train)
runtime_dt = time.time() - start_time

best_dt = grid_search_dt.best_estimator_
y_proba_dt = evaluate_model(best_dt, X_test, y_test)

print(f"Best Parameters for DT: {grid_search_dt.best_params_}")
print(f"Training Time: {runtime_dt:.2f} seconds\n")

# Compare Feature Importance
dt_importances = best_dt.feature_importances_
dt_indices = np.argsort(dt_importances)[::-1]

plt.figure(figsize=(12, 6))
plt.title("Feature Importances (Decision Tree)")
plt.bar(range(X.shape[1]), dt_importances[dt_indices], align="center")
plt.xticks(range(X.shape[1]), [data.feature_names[i] for i in dt_indices], rotation=90)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()

# Plot ROC Curve for Decision Tree
plt.figure(figsize=(8, 6))
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_proba_dt)

plt.plot(fpr_dt, tpr_dt, label="Decision Tree (AUC = {:.2f})".format(roc_auc_score(y_test, y_proba_dt)))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Decision Tree")
plt.legend()
plt.show()

# Select the 10 most important features
top_n = 10
top_features_dt = [data.feature_names[i] for i in dt_indices[:top_n]]

print("Top 10 Features (Decision Tree):", top_features_dt)

# Create a new dataset with the top 10 features
X_train_dt_top = pd.DataFrame(X_train, columns=data.feature_names)[top_features_dt]
X_test_dt_top = pd.DataFrame(X_test, columns=data.feature_names)[top_features_dt]

# Retrain Decision Tree with the top 10 features
best_dt.fit(X_train_dt_top, y_train)
y_proba_dt_reduced = evaluate_model(best_dt, X_test_dt_top, y_test)

# Compare Performance
plt.figure(figsize=(8, 6))
fpr_dt_reduced, tpr_dt_reduced, _ = roc_curve(y_test, y_proba_dt_reduced)

plt.plot(fpr_dt, tpr_dt, label="DT (All Features) AUC = {:.2f}".format(roc_auc_score(y_test, y_proba_dt)))
plt.plot(fpr_dt_reduced, tpr_dt_reduced, label="DT (Top 10 Features) AUC = {:.2f}".format(roc_auc_score(y_test, y_proba_dt_reduced)))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Comparison)")
plt.legend()
plt.show()



