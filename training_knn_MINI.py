import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, auc

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

# Entraînement du modèle KNN
knn = KNeighborsClassifier(n_neighbors=5)  # 5 voisins par défaut
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
y_scores = knn.predict_proba(X_test)

# Fonction de calcul de la matrice de confusion
def compute_confusion_matrix(y_true, y_pred):
    classes = np.unique(np.concatenate((y_true, y_pred)))
    conf_matrix = {}
    for cls in classes:
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        tn = np.sum((y_true != cls) & (y_pred != cls))
        conf_matrix[cls] = {"TP": tp, "FP": fp, "FN": fn, "TN": tn}
    return conf_matrix

# Calcul de la matrice de confusion
conf_matrix = compute_confusion_matrix(y_test, y_pred)

# Affichage des métriques pour chaque classe
for cls, metrics in conf_matrix.items():
    tp, fp, fn, tn = metrics.values()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) != 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    print(f"Classe {cls} : Accuracy={accuracy:.4f}, Recall={recall:.4f}, Precision={precision:.4f}, FPR={fpr:.4f}, Specificity={specificity:.4f}")

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
