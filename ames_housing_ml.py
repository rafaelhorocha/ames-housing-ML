"""
Projeto: Previsão de Preço de Imóveis
Dataset: Ames Housing (Kaggle)
Modelo: Regressão Linear
Autor: Rafael Rocha

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# 1. CARREGAMENTO DO DATASET
# -----------------------------
dataset = pd.read_csv("AmesHousing.csv")

print("Dataset carregado com sucesso!")
print(dataset.head())
print("\nDimensão do dataset:", dataset.shape)
print("\nTipos de dados:")
print(dataset.dtypes.value_counts())

# -----------------------------
# 2. ANÁLISE EXPLORATÓRIA
# -----------------------------
plt.figure(figsize=(8, 5))
sns.histplot(dataset["SalePrice"], bins=40, kde=False)
plt.title("Distribuição do Preço dos Imóveis")
plt.xlabel("Preço")
plt.ylabel("Frequência")
plt.tight_layout()
plt.show()

# -----------------------------
# 3. LIMPEZA DOS DADOS
# -----------------------------
# Remove colunas com muitos valores nulos
dataset = dataset.dropna(axis=1, thresh=dataset.shape[0] * 0.7)

# Remove linhas com valores nulos restantes
dataset = dataset.dropna()

# -----------------------------
# 4. SEPARAÇÃO DE VARIÁVEIS
# -----------------------------
X = dataset.drop("SalePrice", axis=1)
y = dataset["SalePrice"]

# -----------------------------
# 5. ENCODING (One-Hot)
# -----------------------------
X = pd.get_dummies(X, drop_first=True)

print("\nDimensão após encoding:", X.shape)

# -----------------------------
# 6. TREINO E TESTE
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 7. TREINAMENTO DO MODELO
# -----------------------------
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# -----------------------------
# 8. AVALIAÇÃO
# -----------------------------
y_pred = modelo.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n📊 RESULTADOS DO MODELO")
print(f"RMSE: {rmse:,.2f}")
print(f"R² Score: {r2:.4f}")

# -----------------------------
# 9. GRÁFICO REAL vs PREVISTO
# -----------------------------
plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    '--r'
)
plt.xlabel("Preço Real")
plt.ylabel("Preço Previsto")
plt.title("Preço Real vs Previsto")
plt.tight_layout()
plt.show()

# -----------------------------
# 10. EXEMPLO DE PREVISÃO
# -----------------------------
exemplo = X_test.iloc[[0]]
previsao = modelo.predict(exemplo)

print("\n🏠 Exemplo de previsão:")
print(f"Preço previsto: R$ {previsao[0]:,.2f}")
