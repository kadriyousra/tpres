import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier

# Chargement du dataset MNIST
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist["data"], mnist["target"].astype(int)

# Normalisation des données
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Division en données d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Initialisation et entraînement du modèle
model = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(25,), max_iter=200, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_scores = model.predict_proba(X_test)

# Calcul de la matrice de confusion
conf_matrix = confusion_matrix(y_test, y_pred)

# Affichage du rapport de classification
print(classification_report(y_test, y_pred))

# Courbe ROC et calcul de l'AUC
def plot_roc_curve(y_true, y_scores, num_classes):
    plt.figure(figsize=(10, 8))
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Classe {i} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Courbe ROC pour chaque classe')
    plt.legend()
    plt.show()

# Tracer la courbe ROC
plot_roc_curve(y_test, y_scores, num_classes=len(np.unique(y)))
