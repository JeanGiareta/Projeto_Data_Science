# Importando bibliotecas
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor  # Importando XGBoost

# Carregando os dados de treino do repositório no GitHub
train_data_url = 'https://raw.githubusercontent.com/JeanGiareta/Projeto_Data_Science/main/Arquivos/train.csv'
train_data = pd.read_csv(train_data_url)

# Carregando os dados de teste do repositório no GitHub
test_data_url = 'https://raw.githubusercontent.com/JeanGiareta/Projeto_Data_Science/main/Arquivos/test.csv'
test_data = pd.read_csv(test_data_url)

# Separando as features e o target
X = train_data.drop('SalePrice', axis=1)
y = train_data['SalePrice']

# Separando os dados em treino e teste
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

# Selecionando colunas numéricas apenas
numeric_cols = [cname for cname in X_train.columns if X_train[cname].dtype in ['int64', 'float64']]
X_train_numeric = X_train[numeric_cols].copy()
X_valid_numeric = X_valid[numeric_cols].copy()

# Tratando valores ausentes
my_imputer = SimpleImputer(strategy='mean')
X_train_numeric_imputed = pd.DataFrame(my_imputer.fit_transform(X_train_numeric))
X_valid_numeric_imputed = pd.DataFrame(my_imputer.transform(X_valid_numeric))

# Ajustando o modelo XGBoost
#model = XGBRegressor(random_state=0)  # Agora estamos usando XGBRegressor

# Ajustando o modelo XGBoost com parâmetros personalizados
model = XGBRegressor(
    n_estimators=1000,  # número de árvores
    learning_rate=0.05,  # taxa de aprendizado
    max_depth=5,  # profundidade máxima da árvore
    subsample=0.8,  # fração de instâncias amostradas aleatoriamente
    colsample_bytree=0.8,  # fração de colunas amostradas aleatoriamente
    gamma=0.1,  # parâmetro de regularização
    reg_alpha=0.1,  # parâmetro de regularização L1
    reg_lambda=0.1,  # parâmetro de regularização L2
    random_state=0
)

# Treinando o modelo
model.fit(X_train_numeric_imputed, y_train)

# Fazendo previsões nos dados de validação
predictions = model.predict(X_valid_numeric_imputed)

# Avaliando o modelo
mse = mean_squared_error(y_valid, predictions)
rmse = mse**0.5
print(f'Root Mean Squared Error: {rmse}')

# Fazendo previsões nos dados de teste
test_numeric = test_data[numeric_cols].copy()
test_numeric_imputed = pd.DataFrame(my_imputer.transform(test_numeric))
test_predictions = model.predict(test_numeric_imputed)

# Criando DataFrame de submissão
submission_df = pd.DataFrame({'Id': test_data['Id'], 'SalePrice': test_predictions})

# Salvando o arquivo de submissão
submission_df.to_csv('submission.csv', index=False, header=True)

# Restante do código para visualização (scatter plot, gráfico de resíduos, histograma) permanece inalterado.

import matplotlib.pyplot as plt

plt.scatter(y_valid, predictions)
plt.title('Gráfico de Dispersão: Valores Reais vs. Previsões')
plt.xlabel('Valores Reais')
plt.ylabel('Previsões')
plt.show()


residuals_dt = y_valid - predictions
plt.scatter(predictions, residuals_dt)
plt.title('Gráfico de Resíduos')
plt.xlabel('Previsões')
plt.ylabel('Resíduos')
plt.axhline(y=0, color='r', linestyle='--')
plt.show()
plt.hist(residuals_dt, bins=30)
plt.title('Histograma dos Resíduos')
plt.xlabel('Resíduos')
plt.ylabel('Frequência')
plt.show()
