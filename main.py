# %% [markdown]
# Passo a Passo:
# 0. Entender o desafio proposto
#     - No nosso caso, temos de analisar e fazer uma previsão se os novos clientes tem um perfil que possibilite a liberação de crédito
# 
# 1. Importar a base de dados
# 
# 2. Preparar a base de dados
#     - As inteligencias artificiais, só concedem entender e estudar valores, então temos de adptar a base de dados, para que uma IA possa ler
# 
# 3. Criar o Modelo de IA --> Score crédito: Good, Standart, poor
# 4. Escolher o modelo mais eficiente para esta tarefa
# 5. Fazer as previsões para os novos clientes
# 

# %%
# 1. Importar a base de dados
import pandas as pd

tabela = pd.read_csv("clientes.csv")


# %%
# 2. Preparar a base de dados
# usando o display.info, eu vi quais colunas não são numeros, então vou usar o labelEncoder nelas para poder convertelas(exeto o score de credito)
from sklearn.preprocessing import LabelEncoder

codificador = LabelEncoder()

colunasAConverter = ['profissao', 'mix_credito', 'comportamento_pagamento']
for coluna in colunasAConverter:
    tabela[coluna] = codificador.fit_transform(tabela[coluna])



# %%
# dados de X e Y <-- X se refere ao material de estudo, e Y a resposta que sera encontrada
y = tabela['score_credito']
colunas = [
    'score_credito',
    'id_cliente'
]

x = tabela.drop(columns=colunas)

from sklearn.model_selection import train_test_split
x_Treino, x_Teste, y_Treino, y_Teste = train_test_split(x, y)

# %%
# 3. Criar o Modelo de IA --> Score crédito: Good, Standart, poor
# Arvore de decisão
from sklearn.ensemble import RandomForestClassifier
# knn
from sklearn.neighbors import KNeighborsClassifier

modelo_arvoredecisão = RandomForestClassifier()
modelo_knn = KNeighborsClassifier()

# Treinando a ia
modelo_arvoredecisão.fit(x_Treino, y_Treino)
modelo_knn.fit(x_Treino, y_Treino)

# %%
# 4. Escolher o melhor modelo
from sklearn.metrics import accuracy_score
previsao_arvoredecisao = modelo_arvoredecisão.predict(x_Teste)
previsao_knn = modelo_knn.predict(x_Teste.to_numpy()) # so tem esse numpy pelo knn precisar dos dados convertidos para este formato

print(accuracy_score(y_Teste, previsao_arvoredecisao))
print(accuracy_score(y_Teste, previsao_knn))


# %% [markdown]
# 
# 
# Para este caso o modelo Arvore de decisão teve uma acuracia melhor, tendo uma taxa de acerto de 82% aproximadamente, sendo que o do neighboor, teve de 74%



# %%
# 5. fazendo novas previsões
novos_clientes = pd.read_csv("novos_clientes.csv")
for coluna in novos_clientes.columns:
    if novos_clientes[coluna].dtype == "object" and coluna != "score_credito":
        novos_clientes[coluna] = codificador.fit_transform(novos_clientes[coluna])

previsoes = modelo_arvoredecisão.predict(novos_clientes)
print(previsoes)

# %%
