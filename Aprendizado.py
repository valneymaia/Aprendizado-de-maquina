
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt


# Atualize o caminho do arquivo para o novo arquivo na área de trabalho
file_path = 'C:\\Users\\W10\\Desktop\\dados.xlsx'
df = pd.read_excel(file_path)

# Exclui colunas não necessárias e define variáveis dependentes
dados = df.values
x = np.delete(dados, [0, 3, 4], axis=1)  # Remove as colunas "Rodada", "GS2T" e "GF2T"
y = dados[:, [3, 4]]  # GS2T e GF2T como variáveis dependentes

# Engenharia de atributos - Normalização
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Divisão do conjunto de dados em treino e teste, mantendo os índices originais
x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(
    x_scaled, y, df.index, test_size=0.2, random_state=42)

# Adiciona interações polinomiais
poly = PolynomialFeatures(degree=2)
x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.transform(x_test)

# Modelo de Regressão Linear com polinômios
model = LinearRegression()
model.fit(x_train_poly, y_train)
y_pred = model.predict(x_test_poly)

# Função para arredondar as previsões conforme solicitado
def arredondar_previsao(valor):
    if valor >= 1.50:
        return 2
    elif valor >= 0.50:
        return 1
    else:
        return 0

# Imprime as previsões arredondadas e os valores reais, junto com mais dados das partidas
print("\nPrevisões e Valores Reais (GF2T e GS2T) com arredondamento:")
for i in range(len(y_pred)):
    previsao_gf2t = arredondar_previsao(y_pred[i][1])
    previsao_gs2t = arredondar_previsao(y_pred[i][0])
    
    real_gf2t = y_test[i][1]
    real_gs2t = y_test[i][0]

    # Pegamos os dados da rodada correta usando o índice original
    idx = idx_test[i]
    rodada = df.loc[idx, 'Rodada']
    posse_fla = df.loc[idx, 'PosseFla']
    escanteios_fla = df.loc[idx, 'EscanteiosFla']
    finalizacoes_fla = df.loc[idx, 'FinaTotal']
    
    print(f"Rodada {rodada} | Posse de Bola Fla: {posse_fla}% | Escanteios Fla: {escanteios_fla} | Finalizações Fla: {finalizacoes_fla}")
    
    # Comparação das previsões arredondadas com os valores reais
    print(f"Previsão - GF2T: {previsao_gf2t}, GS2T: {previsao_gs2t} | Valor Real - GF2T: {real_gf2t}, GS2T: {real_gs2t}")
    
    # Exibe "green" se as previsões forem iguais aos valores reais
    if previsao_gf2t == real_gf2t and previsao_gs2t == real_gs2t:
        print("Resultado: GREEN")

# Métricas de desempenho do modelo original
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
y_test_flat = y_test.flatten()
y_pred_flat = y_pred.flatten()
pearson_corr = np.corrcoef(y_test_flat, y_pred_flat)[0, 1]

print(f"\nMean Squared Error: {mse}")
print(f'R^2 Score: {r2}')
print(f'Pearson Correlation Coefficient: {pearson_corr}')


