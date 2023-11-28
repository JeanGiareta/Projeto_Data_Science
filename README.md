# Projeto de Data Science

## Objetivo
O objetivo deste projeto é desenvolver um modelo de previsão de preços de imóveis usando XGBoost. 
O conjunto de dados utilizado e descrição do desafio é encontrado em:
https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview

## Conteúdo
- **Pasta Raiz:**
  - Este diretório concentra todos os arquivos do projeto.

- **Arquivos:**
  - A pasta 'Arquivos' armazena os dataSets utilizados para treinamento e teste do modelo

- **src:**
  - A pasta 'scr'armazena o codigo fonte do projeto
 
- **Relatório:**
  - O arquivo 'README.md' é o relatório final do projeto, apresentando detalhes sobre o tema, URL no GitHub, dataset utilizado, modelos desenvolvidos e resultados obtidos, o mesmo fica na raiz.

### Origem
Os dados foram obtidos do seguinte repositório no GitHub:
- [Dados de Treino](https://raw.githubusercontent.com/JeanGiareta/Projeto_Data_Science/main/Arquivos/train.csv)
- [Dados de Teste](https://raw.githubusercontent.com/JeanGiareta/Projeto_Data_Science/main/Arquivos/test.csv)

### Variáveis
O conjunto de dados contém várias variáveis numericas que foram selecionadas, incluindo:
- `LotArea`: Tamanho do lote em pés quadrados
- `GarageArea`: Tamanho da garagem em pés quadrados

### Transformações Realizadas
Ao carregar e preparar os dados, as seguintes etapas foram executadas:

1. **Separação de Features e Target:**
   - `X`: Conjunto de features.
   - `y`: Variável alvo (`SalePrice`).

2. **Divisão dos Dados em Treino e Teste:**
   Os dados foram divididos em conjuntos de treino (80%) e teste (20%) usando a função `train_test_split`.

3. **Seleção de Colunas Numéricas:**
   Foram selecionadas apenas as colunas numéricas para análise.

4. **Tratamento de Valores Ausentes:**
   - Utilização do SimpleImputer para preencher valores ausentes utilizando a média.
  
## Modelo XGBoost
### Ajuste do Modelo
O modelo XGBoost foi ajustado com os seguintes parâmetros personalizados:

```python
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
```
## Treinamento do Modelo
O modelo foi treinado utilizando os dados de treino após o pré-processamento.

## Resultados
O desempenho do modelo foi avaliado nos dados de validação usando a métrica Root Mean Squared Error (RMSE).

### Gráfico de Dispersão: Valores Reais vs. Previsões
O gráfico de dispersão representa a comparação entre os valores reais e as previsões do modelo nos dados de validação.

### Gráfico de Resíduos
O gráfico de resíduos fornece insights sobre a distribuição dos erros do modelo. O eixo x representa as previsões, o eixo y representa os resíduos.

### Histograma dos Resíduos
O histograma dos resíduos mostra a distribuição dos erros do modelo.

## Configuração do Ambiente
Antes de começar, certifique-se de ter a biblioteca XGBoost instalada. Se ainda não estiver instalada, você pode fazê-lo usando o comando:
```bash
pip install xgboost
````
## Como Utilizar Este Repositório

1. **Clonar o Repositório:**
   ```bash
   git clone https://github.com/JeanGiareta/Projeto_Data_Science.git
   cd Projeto_Data_Science

2. **Executar o Script:**
   ```bash
   python src/xgboost.py

## Contribuições

Contribuições e feedbacks são bem-vindas! Sinta-se à vontade para abrir issues e propor melhorias.

Autor:
Jean Francisco Giareta

Contato:
172959@upf.br
