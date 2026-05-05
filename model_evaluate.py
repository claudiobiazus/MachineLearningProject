import os
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report

from sklearn.pipeline import Pipeline

### Modelos que serão avaliados
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV

# # -------------------------
# # Path do dataset para ser avaliado nos modelos
# # -------------------------
PATH = "/home/pedro/Datasets/FER"
TRAIN = PATH + "/train_features_model.csv"
TEST = PATH + "/test_features_model.csv"

def train_and_evaluate(X_train, y_train, X_test, y_test, model, param_grid):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    param_grid = {
        f"model__{key}": value for key, value in param_grid.items()
    }
    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    ## Grid search usando o pipeline, utiliza o scaler de maneira correta
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=skf,
        scoring='f1_macro',
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    print("\nMelhores parâmetros encontrados:")
    print(grid.best_params_)

    print("Score médio (CV):")
    print(grid.best_score_)

    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)
    report = classification_report(y_test, y_pred)

    result_text = ""
    result_text += "Parâmetros:\n"
    result_text += str(grid.best_params_) + "\n\n"

    result_text += "Score médio (CV):\n"
    result_text += str(grid.best_score_) + "\n\n"

    result_text += "Reports:\n"
    result_text += report + "\n"

    return result_text


models_and_parameters = [
    # {
    #     "name": "LogisticRegression",
    #     "model": LogisticRegression(max_iter=1000),
    #     "params": {
    #         "C": [0.01, 0.1, 1, 10]
    #     }
    # },
    #
    # {
    #     "name": "KNN",
    #     "model": KNeighborsClassifier(),
    #     "params": {
    #         "n_neighbors": [3, 5, 7, 9]
    #     }
    # },
    # {
    #     "name": "SVM",
    #     "model": SVC(),
    #     "params": {
    #         "C": [0.1, 1, 10],
    #         "kernel": ["rbf", "linear"]
    #     }
    # },
    # {
    #     "name": "MLP",
    #     "model": MLPClassifier(max_iter=1000),
    #     "params": {
    #         "hidden_layer_sizes": [(50,), (50, 50)],
    #         "alpha": [0.0001, 0.001]
    #     }
    # },
    # {
    #     "name": "RandomForest",
    #     "model": RandomForestClassifier(),
    #     "params": {
    #         "n_estimators": [100, 200],
    #         "max_depth": [None, 5, 10]
    #     }
    # },
    {
        "name": "DecisionTree",
        "model": DecisionTreeClassifier(),
        "params": {
            "max_depth": [None, 5, 10]
        }
    },
    # {
    #     "name": "NaiveBayes",
    #     "model": GaussianNB(),
    #     "params": {
    #         # vazio mesmo
    #     }
    # },
    # {
    #     "name": "LDA",
    #     "model": LinearDiscriminantAnalysis(),
    #     "params": {
    #         # vazio mesmo
    #     }
    # }
]

DATASET = TRAIN.split("/")[-1]

# Carregar dados
X_train = pd.read_csv(TRAIN)
y_train = X_train['label']
X_train = X_train.drop('label', axis=1)

X_test = pd.read_csv(TEST)
y_test = X_test['label']
X_test = X_test.drop('label', axis=1)

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print("TESTE DE CLASSIFICADORES")
print(f"{DATASET}\n")

for item in models_and_parameters:
    print("\nMODELO:", item["name"])

    result_text = train_and_evaluate(
        X_train, y_train,
        X_test, y_test,
        item["model"],
        item["params"]
    )

    file_path = os.path.join(OUTPUT_DIR, f"{item['name']}.txt")
    with open(file_path, "w") as f:
        f.write(result_text)