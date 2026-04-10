# model_analysis.py
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix

# -------------------------------
# 1 - Feature Importance
# -------------------------------
def plot_feature_importance(model, feature_names, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    plt.figure(figsize=(10,6))
    plt.bar(sorted_features, sorted_importances)
    plt.xticks(rotation=45, ha='right')
    plt.title("Importância das Features (Random Forest)")
    plt.ylabel("Importância")
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, "feature_importance.png"), dpi=300)
    plt.close()


# -------------------------------
# 2 - Matriz de Confusão
# -------------------------------
def plot_confusion_matrix(y_true, y_pred, model_name, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Matriz de Confusão - {model_name}")
    plt.xlabel("Predito")
    plt.ylabel("Real")

    plt.savefig(os.path.join(output_dir, f"confusion_matrix_{model_name}.png"), dpi=300)
    plt.close()


# -------------------------------
# 3 - Curva ROC + AUC
# -------------------------------
def plot_all_roc(models, X_test, y_test, output_dir="plots"):
    plt.figure(figsize=(8,6))

    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:,1]
        else:
            y_prob = model.decision_function(X_test)

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("Comparação de Curvas ROC")
    plt.legend()

    plt.savefig(os.path.join(output_dir, "roc_comparacao.png"), dpi=300)
    plt.close()