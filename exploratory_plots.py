# exploratory_plots.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def exploratory_analysis(df, output_dir="plots"):
    """
    Cria e salva gráficos de análise exploratória do dataset Titanic.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Heatmap de valores ausentes
    plt.figure(figsize=(10,6))
    sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
    plt.savefig(os.path.join(output_dir, "heatmap_valores_ausentes.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Gráficos de contagem
    count_plots = [
        ('Pclass', 'Sobrevivência x Classe', 'Classe', 'Número de Pessoas', 'sobrevivencia_por_classe.png'),
        ('Sex', 'Sobrevivência x Sexo', 'Sexo', 'Número de Pessoas', 'sobrevivencia_por_sexo.png'),
        ('Embarked', 'Sobrevivência x Porto de Embarque', 'Porto de Embarque', 'Número de Pessoas', 'sobrevivencia_por_embarque.png')
    ]

    for col, title, xlabel, ylabel, fname in count_plots:
        plt.figure(figsize=(8,6))
        sns.countplot(x=col, hue='Survived', data=df)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(os.path.join(output_dir, fname), dpi=300, bbox_inches='tight')
        plt.close()

    # Distribuição de Fare e Age
    dist_plots = [
        ('Fare', 'Distribuição da Fare por Sobrevivência', 'Fare', 'Contagem', 'fare_por_sobrevivencia.png'),
        ('Age', 'Distribuição da Idade por Sobrevivência', 'Idade', 'Contagem', 'idade_por_sobrevivencia.png')
    ]

    for col, title, xlabel, ylabel, fname in dist_plots:
        plt.figure(figsize=(8,6))
        sns.histplot(data=df, x=col, hue='Survived', bins=30, kde=True)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(os.path.join(output_dir, fname), dpi=300, bbox_inches='tight')
        plt.close()

    # Taxa de sobrevivência (%) por Classe
    taxa_sobrevivencia = df.groupby('Pclass')['Survived'].mean() * 100
    plt.figure(figsize=(8,6))
    sns.barplot(x=taxa_sobrevivencia.index, y=taxa_sobrevivencia.values)
    plt.xlabel('Classe do Passageiro (Pclass)')
    plt.ylabel('Taxa de Sobrevivência (%)')
    plt.title('Taxa de Sobrevivência por Classe do Passageiro')
    plt.savefig(os.path.join(output_dir, 'taxa_sobrevivencia_por_classe.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Taxa de sobrevivência (%) por faixa etária
    df_temp = df.copy()
    df_temp['Age'] = df_temp['Age'].fillna(df_temp['Age'].median())

    df_temp['Age_Group'] = pd.cut(df_temp['Age'],
                                  bins=[0, 5, 10, 15, 20, 30, 40, 50, 100],
                                  labels=['0-4', '5-9', '10-14', '15-19', '20-29', '30-39', '40-49', '50+'])

    survival_by_age = df_temp.groupby('Age_Group', observed=True)['Survived'].mean() * 100

    plt.figure(figsize=(8, 6))
    sns.barplot(x=survival_by_age.index, y=survival_by_age.values)
    plt.xlabel('Faixa Etária')
    plt.ylabel('Taxa de Sobrevivência (%)')
    plt.title('Taxa de Sobrevivência por Faixa Etária')
    plt.savefig(os.path.join(output_dir, 'taxa_sobrevivencia_por_idade.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Porcentagem de valores ausentes
    missing_percentage = (df.isna().sum() / df.shape[0] * 100).sort_values(ascending=False)
    print("\nPorcentagem de valores ausentes por coluna (%):")
    print(missing_percentage)

    print(f"\nTodos os gráficos foram salvos na pasta '{output_dir}'.")