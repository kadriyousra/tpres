import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import random
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve
from sklearn.preprocessing import StandardScaler

# Load data
X = np.genfromtxt('data.csv', delimiter=',', dtype=int)
Y = np.genfromtxt('labels.csv', delimiter=',', dtype=int)

# Convert labels 10 to 0
Y = np.where(Y == 10, 0, Y)

# Visualize random samples
def plot_random_samples(X, Y, n_samples=13):
    plt.figure(figsize=(15,8))
    for i in range(n_samples):
        c = random.randint(X.shape[0])
        a = X[c,:].reshape((20, 20))
        a = np.transpose(a)
        plt.subplot(1,n_samples,i+1)
        plt.title('label '+ str(Y[c]))
        plt.imshow(a,cmap='gray')
    plt.show()

# Metric computation functions
def compute_confusion_matrix(y_true, y_pred):
    if y_true.shape != y_pred.shape:
        raise ValueError("Les dimensions de y_true et y_pred ne correspondent pas !")
    
    classes = np.unique(np.concatenate((y_true, y_pred)))
    conf_matrix = {}
    
    for cls in classes:
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        tn = np.sum((y_true != cls) & (y_pred != cls))
        
        conf_matrix[cls] = {"TP": tp, "FP": fp, "FN": fn, "TN": tn}
    
    return conf_matrix

def compute_metrics(tp, tn, fp, fn):
    metrics = {
        "accuracy": (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0,
        "recall": tp / (tp + fn) if (tp + fn) != 0 else 0,
        "precision": tp / (tp + fp) if (tp + fp) != 0 else 0,
        "fpr": fp / (fp + tn) if (fp + tn) != 0 else 0,
        "specificity": tn / (tn + fp) if (tn + fp) != 0 else 0
    }
    return metrics

def plot_roc_curve(y_true, y_scores, num_classes):
    plt.figure(figsize=(10, 8))
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')  # Random guess line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

# Data preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, Y, test_size=0.3, random_state=42
)

# Train decision tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Make predictions
y_pred = dt_model.predict(X_test)
y_pred_proba = dt_model.predict_proba(X_test)

# Evaluate model
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot ROC curve
plot_roc_curve(y_test, y_pred_proba, num_classes=10)