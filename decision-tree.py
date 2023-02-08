import pandas as pd
from datetime import datetime

uri = "https://gist.githubusercontent.com/guilhermesilveira/4d1d4a16ccbf6ea4e0a64a38a24ec884/raw/afd05cb0c796d18f3f5a6537053ded308ba94bf7/car-prices.csv"

datas = pd.read_csv(uri)

swap = {
  'yes' : 1,
  'no' : 0
}
datas.sold = datas.sold.map(swap)

current_year = datetime.today().year # armazenando o ano atual
datas['model_age'] = current_year - datas.model_year # calculando a idade do modelo
datas['kilometers_per_year'] = datas.mileage_per_year * 1.60934 # convertendo milhas para km

datas = datas.drop(columns = ["Unnamed: 0", "mileage_per_year", "model_year"])

import numpy as np
from sklearn. preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

x = datas[["price", "model_age", "kilometers_per_year"]]
y = datas["sold"]

SEED = 5
np.random.seed(SEED)

raw_train_x, raw_test_x, train_y, test_y = train_test_split(x, y, test_size = 0.25,
stratify = y)

scaler = StandardScaler()
scaler.fit(raw_train_x)
train_x = scaler.transform(raw_train_x) 
test_x = scaler.transform(raw_test_x) 

print(f"Treinaremos com {len(train_x)} elementos e testaremos com {len(test_x)} elementos")

model = SVC(gamma = 'auto')
model.fit(train_x, train_y) 
predict = model.predict(test_x) 

accuracy = accuracy_score(test_y, predict) * 100
print("A acurácia foi de %.2f%%" % accuracy)

# criando baseline com dummy stratified
from sklearn.dummy import DummyClassifier

dummy_stratified = DummyClassifier()
dummy_stratified.fit(train_x, train_y)
accuracy = dummy_stratified.score(test_x, test_y) * 100

print("A acuracia do dummy stratified foi de: %.2f%%" % accuracy)

# criando baseline com dummy mostfrequent
dummy_mostfrequent = DummyClassifier()
dummy_mostfrequent.fit(train_x, train_y)
accuracy = dummy_mostfrequent.score(test_x, test_y) * 100

print("A acuracia do dummy mostfrequent foi de: %.2f%%" % accuracy)
