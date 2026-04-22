from main import preprocess, load_and_inspect, exploratory_analysis

from sklearn.tree import DecisionTreeClassifier
import pandas as pd

CSV_PATH = "/home/pedro/Datasets/Titanic-Dataset.csv"
df = load_and_inspect(CSV_PATH) # 1 - Carregar e inspecionar
X_train, X_test, y_train, y_test, _ = preprocess(df) # 3 - Pré-processamento

# treinar modelo
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# extrair importância
importances = model.feature_importances_

# organizar bonitinho
feat_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': importances
}).sort_values(by='importance', ascending=False)