
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
from sklearn.feature_selection import SelectKBest, f_classif
import time
import warnings

warnings.filterwarnings('ignore')
# Load dataset
data = load_breast_cancer()
X = data.data  # Features
y = data.target  # Labels (0 = malignant, 1 = benign)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Function to evaluate models
def evaluate_model(model, X_test, y_test, model_name=None):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    if model_name:
        print(f"Evaluation for {model_name}:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"ROC AUC: {roc_auc:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    return y_proba

# ===== Decision Tree Model =====
print("\n" + "="*50)
print("Decision Tree Classifier")
print("="*50)

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
y_proba_dt = evaluate_model(best_dt, X_test, y_test, "Decision Tree")

print(f"Best Parameters for DT: {grid_search_dt.best_params_}")
print(f"Training Time: {runtime_dt:.2f} seconds\n")

# ===== KNN Model =====
print("\n" + "="*50)
print("K-Nearest Neighbors Classifier")
print("="*50)

param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}
knn_model = KNeighborsClassifier()
grid_search_knn = GridSearchCV(estimator=knn_model, param_grid=param_grid_knn, scoring='accuracy', cv=5, verbose=1, n_jobs=-1)

start_time = time.time()
grid_search_knn.fit(X_train, y_train)
runtime_knn = time.time() - start_time

best_knn = grid_search_knn.best_estimator_
y_proba_knn = evaluate_model(best_knn, X_test, y_test, "KNN")

print(f"Best Parameters for KNN: {grid_search_knn.best_params_}")
print(f"Training Time: {runtime_knn:.2f} seconds\n")

# ===== Random Forest Model =====
print("\n" + "="*50)
print("Random Forest Classifier")
print("="*50)

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
y_proba_rf = evaluate_model(best_rf, X_test, y_test, "Random Forest")

print(f"Best Parameters for RF: {grid_search_rf.best_params_}")
print(f"Training Time: {runtime_rf:.2f} seconds\n")

# ===== XGBoost Model =====
print("\n" + "="*50)
print("XGBoost Classifier")
print("="*50)

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
y_proba_xgb = evaluate_model(best_xgb, X_test, y_test, "XGBoost")

print(f"Best Parameters for XGB: {grid_search_xgb.best_params_}")
print(f"Training Time: {runtime_xgb:.2f} seconds\n")

# ===== Feature Importance Analysis =====
print("\n" + "="*50)
print("Feature Importance Analysis")
print("="*50)

# Get feature importances for each model (except KNN)
dt_importances = best_dt.feature_importances_
rf_importances = best_rf.feature_importances_
xgb_importances = best_xgb.feature_importances_

# Sort indices
dt_indices = np.argsort(dt_importances)[::-1]
rf_indices = np.argsort(rf_importances)[::-1]
xgb_indices = np.argsort(xgb_importances)[::-1]

# Get top 10 features for each model
top_n = 10
top_features_dt = [data.feature_names[i] for i in dt_indices[:top_n]]
top_features_rf = [data.feature_names[i] for i in rf_indices[:top_n]]
top_features_xgb = [data.feature_names[i] for i in xgb_indices[:top_n]]

print("Top 10 Features (Decision Tree):", top_features_dt)
print("Top 10 Features (Random Forest):", top_features_rf)
print("Top 10 Features (XGBoost):", top_features_xgb)

# For KNN, use SelectKBest for feature selection
selector = SelectKBest(score_func=f_classif, k=10)
selector.fit(X_train, y_train)
selected_features_knn = np.array(data.feature_names)[selector.get_support()]
print("Top 10 Features (KNN - SelectKBest):", selected_features_knn.tolist())

# ===== Plotting Feature Importances =====
# Plot Decision Tree feature importances
plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1)
plt.title("Feature Importances (Decision Tree)")
plt.bar(range(top_n), dt_importances[dt_indices[:top_n]], align="center")
plt.xticks(range(top_n), [data.feature_names[i] for i in dt_indices[:top_n]], rotation=90)
plt.tight_layout()

# Plot Random Forest feature importances
plt.subplot(2, 2, 2)
plt.title("Feature Importances (Random Forest)")
plt.bar(range(top_n), rf_importances[rf_indices[:top_n]], align="center")
plt.xticks(range(top_n), [data.feature_names[i] for i in rf_indices[:top_n]], rotation=90)
plt.tight_layout()

# Plot XGBoost feature importances
plt.subplot(2, 2, 3)
plt.title("Feature Importances (XGBoost)")
plt.bar(range(top_n), xgb_importances[xgb_indices[:top_n]], align="center", color='orange')
plt.xticks(range(top_n), [data.feature_names[i] for i in xgb_indices[:top_n]], rotation=90)
plt.tight_layout()

# Plot KNN feature selection scores
plt.subplot(2, 2, 4)
plt.title("Feature Selection Scores (KNN)")
knn_scores = selector.scores_[selector.get_support()]
knn_features = np.array(data.feature_names)[selector.get_support()]
# Sort by score
knn_sorted_idx = np.argsort(knn_scores)[::-1]
plt.bar(range(len(knn_scores)), knn_scores[knn_sorted_idx], align="center", color='green')
plt.xticks(range(len(knn_scores)), knn_features[knn_sorted_idx], rotation=90)
plt.tight_layout()

plt.show()

# ===== Reduced Feature Sets =====
print("\n" + "="*50)
print("Reduced Feature Set Performance")
print("="*50)

# Create reduced feature datasets
X_train_dt_top = pd.DataFrame(X_train, columns=data.feature_names)[top_features_dt]
X_test_dt_top = pd.DataFrame(X_test, columns=data.feature_names)[top_features_dt]

X_train_knn_top = pd.DataFrame(X_train, columns=data.feature_names)[selected_features_knn]
X_test_knn_top = pd.DataFrame(X_test, columns=data.feature_names)[selected_features_knn]

X_train_rf_top = pd.DataFrame(X_train, columns=data.feature_names)[top_features_rf]
X_test_rf_top = pd.DataFrame(X_test, columns=data.feature_names)[top_features_rf]

X_train_xgb_top = pd.DataFrame(X_train, columns=data.feature_names)[top_features_xgb]
X_test_xgb_top = pd.DataFrame(X_test, columns=data.feature_names)[top_features_xgb]

# Retrain models with reduced feature sets
best_dt.fit(X_train_dt_top, y_train)
y_proba_dt_reduced = evaluate_model(best_dt, X_test_dt_top, y_test, "Decision Tree (Top 10 Features)")

best_knn.fit(X_train_knn_top, y_train)
y_proba_knn_reduced = evaluate_model(best_knn, X_test_knn_top, y_test, "KNN (Top 10 Features)")

best_rf.fit(X_train_rf_top, y_train)
y_proba_rf_reduced = evaluate_model(best_rf, X_test_rf_top, y_test, "Random Forest (Top 10 Features)")

best_xgb.fit(X_train_xgb_top, y_train)
y_proba_xgb_reduced = evaluate_model(best_xgb, X_test_xgb_top, y_test, "XGBoost (Top 10 Features)")

# ===== ROC Curve Comparison =====
# Compare all models with full feature set
plt.figure(figsize=(12, 10))
plt.subplot(2, 2, 1)
plt.title("ROC Curves - All Models (Full Features)")

fpr_dt, tpr_dt, _ = roc_curve(y_test, y_proba_dt)
fpr_knn, tpr_knn, _ = roc_curve(y_test, y_proba_knn)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_proba_xgb)

plt.plot(fpr_dt, tpr_dt, label=f"Decision Tree (AUC = {roc_auc_score(y_test, y_proba_dt):.2f})")
plt.plot(fpr_knn, tpr_knn, label=f"KNN (AUC = {roc_auc_score(y_test, y_proba_knn):.2f})")
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {roc_auc_score(y_test, y_proba_rf):.2f})")
plt.plot(fpr_xgb, tpr_xgb, label=f"XGBoost (AUC = {roc_auc_score(y_test, y_proba_xgb):.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()

# Compare Decision Tree with full vs reduced features
plt.subplot(2, 2, 2)
plt.title("ROC Curve - Decision Tree")

fpr_dt_reduced, tpr_dt_reduced, _ = roc_curve(y_test, y_proba_dt_reduced)

plt.plot(fpr_dt, tpr_dt, label=f"DT (All Features) AUC = {roc_auc_score(y_test, y_proba_dt):.2f}")
plt.plot(fpr_dt_reduced, tpr_dt_reduced, label=f"DT (Top 10) AUC = {roc_auc_score(y_test, y_proba_dt_reduced):.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()

# Compare KNN with full vs reduced features
plt.subplot(2, 2, 3)
plt.title("ROC Curve - KNN")

fpr_knn_reduced, tpr_knn_reduced, _ = roc_curve(y_test, y_proba_knn_reduced)

plt.plot(fpr_knn, tpr_knn, label=f"KNN (All Features) AUC = {roc_auc_score(y_test, y_proba_knn):.2f}")
plt.plot(fpr_knn_reduced, tpr_knn_reduced, label=f"KNN (Top 10) AUC = {roc_auc_score(y_test, y_proba_knn_reduced):.2f}")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()

# Compare Random Forest with full vs reduced features
plt.subplot(2, 2, 4)
plt.title("ROC Curve - Random Forest")

fpr_rf_reduced, tpr_rf_reduced, _ = roc_curve(y_test, y_proba_rf_reduced)

plt.plot(fpr_rf, tpr_rf, label=f"RF (All Features) AUC = {roc_auc_score(y_test, y_proba_rf):.2f}")
plt.plot(fpr_rf_reduced, tpr_rf_reduced, label=f"RF (Top 10) AUC = {roc_auc_score(y_test, y_proba_rf_reduced):.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()

plt.tight_layout()
plt.show()
