# main.py
import os
import cv2
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from exploratory_plots import exploratory_analysis

from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import LabelEncoder
from skimage.feature import hog

from model_analysis import plot_feature_importance, plot_confusion_matrix, plot_all_roc
from sklearn.metrics import roc_curve, auc

# -------------------------------
# 0 - Função: Carregar Fer-2013
# -------------------------------
def process_pixels(pixel_str):
    import numpy as np
    pixels = np.array(pixel_str.split(), dtype='float32')
    return pixels / 255.0


def load_fer2013(csv_path):
    df = pd.read_csv(csv_path)

    print("Primeiras linhas:")
    print(df.head())

    X = np.array(df['pixels'].apply(process_pixels).tolist())
    y = df['emotion']

    print("\nShape X:", X.shape)
    print("Shape y:", y.shape)

    return X, y

def load_fer_from_folders(base_path, img_size=(48, 48), limit_per_class=None):
    X = []
    y = []

    class_labels = os.listdir(base_path)
    print("Classes encontradas:", class_labels)

    for label in class_labels:
        class_path = os.path.join(base_path, label)

        if not os.path.isdir(class_path):
            continue

        images = os.listdir(class_path)

        if limit_per_class:
            images = images[:limit_per_class]

        for img_name in images:
            img_path = os.path.join(class_path, img_name)

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            img = cv2.resize(img, img_size)
            img = img.flatten() / 255.0

            X.append(img)
            y.append(label)

    X = np.array(X)
    y = np.array(y)

    le = LabelEncoder()
    y = le.fit_transform(y)

    print("Shape X:", X.shape)
    print("Shape y:", y.shape)

    return X, y

def extract_hog_features(images):
    features = []

    for img in images:
        # garantir formato correto (48x48)
        img = img.reshape(48, 48)

        hog_feat = hog(
            img,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm='L2-Hys',
            visualize=False
        )

        features.append(hog_feat)

    return np.array(features)

# -------------------------------
# 1 - Função: Carregar e inspecionar dados
# -------------------------------
def load_and_inspect(csv_path):
    df = pd.read_csv(csv_path)
    print("Primeiras linhas do dataset:")
    print(df.head())
    print("\nInformações gerais:")
    print(df.info())
    print("\nEstatísticas descritivas:")
    print(df.describe())
    return df

# -------------------------------
# 2 - Função: Pré-processamento
# -------------------------------
def preprocess(df):
    # Valores ausentes
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # Remover colunas irrelevantes
    df.drop(['Cabin', 'PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)

    # Codificação categórica
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

    # Separar features e target
    X = df.drop('Survived', axis=1)
    y = df['Survived']

    # Garantir colunas numéricas
    X = X.select_dtypes(include=['int64', 'float64'])

    # Divisão treino/teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Escalonamento
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("\nPré-processamento concluído!")
    print("Shape X_train:", X_train.shape)
    print("Shape X_test:", X_test.shape)

    return X_train, X_test, y_train, y_test, X.columns

# -------------------------------
# 3 - Função: Treinar e avaliar modelos
# -------------------------------
def train_and_evaluate_models(X_train, X_test, y_train, y_test, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)

    param_grids = {
        "LogisticRegression": {
            "model__C": [0.01, 0.1, 1, 10]
        },
        "KNN": {
            "model__n_neighbors": [3, 5, 7, 9]
        },
        "SVM": {
            "model__C": [0.1, 1, 10],
            "model__kernel": ['rbf', 'linear']
        },
        "MLP": {
            "model__hidden_layer_sizes": [(50,), (50, 50)],
            "model__alpha": [0.0001, 0.001]
        },
        "RandomForest": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 5, 10]
        },
        "DecisionTree": {
            "model__max_depth": [None, 5, 10]
        },
        "NaiveBayes": {},  # não precisa tunar muito
        "LDA": {}  # geralmente já funciona bem
    }

    # -------------------------
    # Modelos com e sem escala
    # -------------------------
    models = {
        "LogisticRegression": Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(max_iter=500))
        ]),
        "KNN": Pipeline([
            ('scaler', StandardScaler()),
            ('model', KNeighborsClassifier(n_neighbors=5))
        ]),
        "SVM": Pipeline([
            ('scaler', StandardScaler()),
            ('model', SVC(probability=True, kernel='rbf', random_state=42))
        ]),
        "MLP": Pipeline([
            ('scaler', StandardScaler()),
            ('model', MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=1000, early_stopping=True, random_state=42))
        ]),
        "LDA": Pipeline([
            ('scaler', StandardScaler()),
            ('model', LinearDiscriminantAnalysis())
        ]),
        "NaiveBayes": Pipeline([
            ('model', GaussianNB())
        ]),
        "RandomForest": Pipeline([
            ('model', RandomForestClassifier(n_estimators=100, random_state=42))
        ]),
        "DecisionTree": Pipeline([
            ('model', DecisionTreeClassifier(random_state=42))
        ])
    }

    results = {}
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Para ROC
    plt.figure(figsize=(8, 6))

    for name, pipeline in models.items():
        # -------------------------
        # Treinar modelo (com ou sem GridSearch)
        # -------------------------
        if param_grids[name]:  # se tiver parâmetros para buscar
            grid = GridSearchCV(
                pipeline,
                param_grids[name],
                cv=kfold,
                scoring='f1',
                n_jobs=-1
            )
            grid.fit(X_train, y_train)

            best_model = grid.best_estimator_
            best_params = grid.best_params_
        else:
            best_model = pipeline
            best_model.fit(X_train, y_train)
            best_params = {}

        # Predição
        y_pred = best_model.predict(X_test)

        # Probabilidades (para ROC)
        if hasattr(best_model, "predict_proba"):
            y_prob = best_model.predict_proba(X_test)[:, 1]
        else:
            y_prob = best_model.decision_function(X_test)

        # -------------------------
        # Métricas
        # -------------------------
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        # -------------------------
        # Cross-validation
        # -------------------------
        cv_acc = cross_val_score(best_model, X_train, y_train, cv=kfold, scoring='accuracy')
        cv_prec = cross_val_score(best_model, X_train, y_train, cv=kfold, scoring='precision_weighted')
        cv_rec = cross_val_score(best_model, X_train, y_train, cv=kfold, scoring='recall_weighted')
        cv_f1 = cross_val_score(best_model, X_train, y_train, cv=kfold, scoring='f1_weighted')

        # -------------------------
        # Resumo Final
        # -------------------------
        results[name] = {
            "cv_accuracy_mean": cv_acc.mean(),
            "cv_accuracy_std": cv_acc.std(),

            "cv_precision_mean": cv_prec.mean(),
            "cv_precision_std": cv_prec.std(),

            "cv_recall_mean": cv_rec.mean(),
            "cv_recall_std": cv_rec.std(),

            "cv_f1_mean": cv_f1.mean(),
            "cv_f1_std": cv_f1.std()
        }

        # -------------------------
        # ROC Curve
        # -------------------------
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

        # -------------------------
        # Salvar resultados
        # -------------------------
        results[name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "cv_accuracy_mean": cv_acc.mean(),
            "cv_precision_mean": cv_prec.mean(),
            "cv_recall_mean": cv_rec.mean(),
            "cv_f1_mean": cv_f1.mean(),
            "auc": roc_auc,
            "best_params": best_params
        }

        # -------------------------
        # Matriz de confusão
        # -------------------------
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Matriz de Confusão - {name}")
        plt.xlabel("Predito")
        plt.ylabel("Real")
        plt.savefig(os.path.join(output_dir, f"confusion_matrix_{name}.png"), dpi=300, bbox_inches='tight')
        plt.close()

    # -------------------------
    # Curva ROC final
    # -------------------------
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Curva ROC - Comparação de Modelos")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=300)
    plt.close()

    # -------------------------
    # Tabela de resultados
    # -------------------------
    results_df = pd.DataFrame(results).T
    results_df = results_df.sort_values(by="auc", ascending=False)

    results_df.to_csv(os.path.join(output_dir, "resultados_modelos.csv"))

    # -------------------------
    # Print organizado
    # -------------------------
    print("\n--- Resultados dos Modelos ---")
    print(results_df)

    return results_df

# -------------------------------
# 4 - FER-2013
# -------------------------------
def train_models_fer_hog(X_raw, y, output_dir="plots_fer"):
    os.makedirs(output_dir, exist_ok=True)

    print("Extraindo HOG features...")
    X = extract_hog_features(X_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "SVM_HOG_PCA": (
            Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA()),
                ('model', SVC(kernel='rbf'))
            ]),
            {
                'pca__n_components': [50, 100, 150],
                'model__C': [1, 5, 10, 20],
                'model__gamma': ['scale', 0.001, 0.0005]
            }
        )
    }

    results = {}

    for name, (pipeline, grid) in models.items():
        print(f"\nTreinando {name}...")

        search = GridSearchCV(
            pipeline,
            grid,
            cv=kfold,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )

        search.fit(X_train, y_train)

        best_model = search.best_estimator_

        y_pred = best_model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        results[name] = {
            "accuracy": acc,
            "f1_score": f1,
            "best_params": search.best_params_
        }

        print(f"{name}")
        print("Accuracy:", acc)
        print("F1:", f1)
        print("Best params:", search.best_params_)

    results_df = pd.DataFrame(results).T
    results_df.to_csv(os.path.join(output_dir, "results_hog_svm.csv"))

    return results_df

# -------------------------------
# 5 - Execução principal
# -------------------------------
if __name__ == "__main__":
    # -------------------------
    # TITANIC
    # -------------------------
    CSV_PATH = "Titanic-Dataset.csv"
    df = load_and_inspect(CSV_PATH) # 1 - Carregar e inspecionar
    exploratory_analysis(df)  # 2 - EDA e gráficos
    X_train, X_test, y_train, y_test, _ = preprocess(df) # 3 - Pré-processamento
    results_titanic = train_and_evaluate_models(X_train, X_test, y_train, y_test) # 4 - Treinar e avaliar modelos

    # -------------------------
    # FER-2013 (HOG + PCA + SVM)
    # -------------------------
    FER_TRAIN_PATH = "archive/train"
    X_raw, y = load_fer_from_folders(FER_TRAIN_PATH, limit_per_class=1000)
    print("Extraindo HOG features...")
    X = extract_hog_features(X_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipeline = Pipeline([
        ('pca', PCA()),
        ('model', SVC(kernel='rbf'))
    ])

    param_grid = {
        'pca__n_components': [50, 100, 150],
        'model__C': [1, 5, 10, 20],
        'model__gamma': ['scale', 0.001, 0.0005]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    svm = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=0
    )

    print("Treinando HOG + PCA + SVM ...")
    svm.fit(X_train, y_train)

    y_pred = svm.predict(X_test)

    print("\nRESULTADOS FINAIS")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1:", f1_score(y_test, y_pred, average='weighted'))
    print("Best params:", svm.best_params_)