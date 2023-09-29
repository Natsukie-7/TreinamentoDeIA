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
display(tabela)

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


