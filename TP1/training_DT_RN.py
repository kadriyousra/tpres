import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, classification_report, roc_curve
from sklearn.preprocessing import StandardScaler

# Chargement des données
X = np.genfromtxt('data.csv', delimiter=',', dtype=int)
Y = np.genfromtxt('labels.csv', delimiter=',', dtype=int)

# Conversion des labels 10 en 0
Y = np.where(Y == 10, 0, Y)

# Prétraitement des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Séparation des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.3, random_state=42)

# Création et entraînement du modèle MLP
model = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(25,), max_iter=100000, random_state=42)
model.fit(X_train, y_train)

# Prédictions sur l'ensemble de test
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Calcul de la matrice de confusion
def compute_confusion_matrix(y_true, y_pred, num_classes):
    conf_matrix = np.zeros((num_classes, 4))  # Chaque ligne contient [TP, FP, FN, TN] pour une classe
    
    for i in range(num_classes):
        tp = np.sum((y_true == i) & (y_pred == i))
        fp = np.sum((y_true != i) & (y_pred == i))
        fn = np.sum((y_true == i) & (y_pred != i))
        tn = np.sum((y_true != i) & (y_pred != i))
        conf_matrix[i] = [tp, fp, fn, tn]
    
    return conf_matrix

num_classes = len(np.unique(y_test))
conf_matrix = compute_confusion_matrix(y_test, y_pred, num_classes)

# Calcul des métriques moyennes
def compute_average_metrics(conf_matrix):
    tp, fp, fn, tn = np.sum(conf_matrix, axis=0)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    recall = np.mean(conf_matrix[:, 0] / (conf_matrix[:, 0] + conf_matrix[:, 2] + 1e-10))
    precision = np.mean(conf_matrix[:, 0] / (conf_matrix[:, 0] + conf_matrix[:, 1] + 1e-10))
    specificity = np.mean(conf_matrix[:, 3] / (conf_matrix[:, 3] + conf_matrix[:, 1] + 1e-10))
    return {"accuracy": accuracy, "recall": recall, "precision": precision, "specificity": specificity}

metrics = compute_average_metrics(conf_matrix)

# Affichage des métriques moyennes
print("\nMétriques moyennes :")
for metric, value in metrics.items():
    print(f"{metric.capitalize()} : {value:.4f}")

# Affichage du rapport de classification
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Tracer la courbe ROC
def plot_roc_curve(y_true, y_scores, num_classes):
    plt.figure(figsize=(10, 8))
    auc_values = []
    
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        auc_values.append(roc_auc)
        plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')
    
    mean_auc = np.mean(auc_values)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve (Mean AUC = {mean_auc:.2f})')
    plt.legend()
    plt.show()

plot_roc_curve(y_test, y_pred_proba, num_classes)
