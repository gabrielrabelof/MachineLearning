import pandas as pd

uri = "https://gist.githubusercontent.com/guilhermesilveira/1b7d5475863c15f484ac495bd70975cf/raw/16aff7a0aee67e7c100a2a48b676a2d2d142f646/projects.csv"

datas = pd.read_csv(uri) # leu os arquivos da internet

swap = {
  0 : 1,
  1 : 0
}

datas['finished'] = datas.unfinished.map(swap) # invertendo valores

x = datas[["expected_hours", "price"]] # separando colunas
y = datas["finished"]                  # para treino e teste

from sklearn. preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import numpy as np

SEED = 5
np.random.seed(SEED)

raw_train_x, raw_test_x, train_y, test_y = train_test_split(x, y, test_size = 0.25,
stratify = y)
# raw = dados de treino e teste crus (originais)

scaler = StandardScaler()
scaler.fit(raw_train_x)
train_x = scaler.transform(raw_train_x) # atribuindo os valores transformados
test_x = scaler.transform(raw_test_x) # ao treino e teste X

print(f"Treinaremos com {len(train_x)} elementos e testaremos com {len(test_x)} elementos")

model = SVC(gamma = 'auto')
model.fit(train_x, train_y) # treinando o algoritmo
predict = model.predict(test_x) # predict

accuracy = accuracy_score(test_y, predict) * 100 # calculando acurácia
print("A acurácia foi de %.2f%%" % accuracy) 

baseline = np.ones(540) # criando uma baseline
accuracy = accuracy_score(test_y, baseline) * 100
print("A acuracia do algoritmo de baseline foi %.2f%%" % accuracy)

data_x = test_x[:,0] # armazenando as horas esperadas e o preço por terem
data_y = test_x[:,1] # se transformado em arrays durante o redimensionamento do gráfico

x_min = data_x.min() # passando por todo
x_max = data_x.max() # o eixo X
y_min = data_y.min() # passando por todo
y_max = data_y.max() # o eixo Y

pixels = 100 # definindo os pixels
eixo_x = np.arange(x_min, x_max, (x_max - x_min) / pixels) # distribuindo
eixo_y = np.arange(y_min, y_max, (y_max - y_min) / pixels) # os pixels

xx, yy = np.meshgrid(eixo_x, eixo_y) # mesclando os pontos
points = np.c_[xx.ravel(), yy.ravel()] # ravel = retorna um array achatado
# np.c_[] = concatenando eixo x com eixo y

Z = model.predict(points) # predict dos novos pontos
Z = Z.reshape(xx.shape) # redimensionando

import matplotlib.pyplot as plt

plt.contourf(xx, yy, Z, alpha = 0.3) # contour = contorno alpha = transparente
plt.scatter(data_x, data_y, c = test_y, s = 1) # scatter = espalhar 
plt.show() # c = cor / s = size

# DECISION BOUNDARY = ?