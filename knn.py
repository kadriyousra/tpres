import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
from sklearn.feature_selection import SelectKBest, f_classif
import time

# Charger le dataset
data = load_breast_cancer()
X = data.data  
y = data.target  

# Diviser en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fonction d'évaluation du modèle
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

# Optimisation des hyperparamètres avec GridSearchCV
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
y_proba_knn = evaluate_model(best_knn, X_test, y_test)

print(f"Best Parameters for KNN: {grid_search_knn.best_params_}")
print(f"Training Time: {runtime_knn:.2f} seconds\n")

# Sélection des caractéristiques avec SelectKBest
selector = SelectKBest(score_func=f_classif, k=10)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

selected_features = np.array(data.feature_names)[selector.get_support()]
scores = selector.scores_[selector.get_support()]

# Afficher les caractéristiques les plus importantes
plt.figure(figsize=(10, 6))
plt.barh(selected_features, scores, color='skyblue')
plt.xlabel("Score d'importance (F-statistic)")
plt.ylabel("Feature")
plt.title("Top 10 Features Importantes pour KNN")
plt.gca().invert_yaxis()  # Inverser l'axe pour afficher la plus importante en haut
plt.show()

print("Top 10 Selected Features:", selected_features)

# Réentraîner KNN avec les 10 meilleures caractéristiques
best_knn.fit(X_train_selected, y_train)
y_proba_knn_reduced = evaluate_model(best_knn, X_test_selected, y_test)

# Comparaison avec la courbe ROC
plt.figure(figsize=(8, 6))
fpr_knn, tpr_knn, _ = roc_curve(y_test, y_proba_knn)
fpr_knn_reduced, tpr_knn_reduced, _ = roc_curve(y_test, y_proba_knn_reduced)

plt.plot(fpr_knn, tpr_knn, label="KNN (All Features) AUC = {:.2f}".format(roc_auc_score(y_test, y_proba_knn)))
plt.plot(fpr_knn_reduced, tpr_knn_reduced, label="KNN (Top 10 Features) AUC = {:.2f}".format(roc_auc_score(y_test, y_proba_knn_reduced)))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (KNN Comparison)")
plt.legend()
plt.show()
