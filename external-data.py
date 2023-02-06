import pandas as pd

uri = "https://gist.githubusercontent.com/guilhermesilveira/2d2efa37d66b6c84a722ea627a897ced/raw/10968b997d885cbded1c92938c7a9912ba41c615/tracking.csv"
datas = pd.read_csv(uri)

x = datas[["home", "how_it_works", "contact"]]
y = datas["bought"]
# print(datas.shape)

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

SEED = 20

train_x, test_x, train_y, test_y = train_test_split(x, y,
random_state = SEED, test_size = 0.25,
stratify = y)

print(f"Treinaremos com {len(train_x)} elementos e testaremos com {len(test_x)} elementos")

model = LinearSVC()
model.fit(train_x, train_y)
predict = model.predict(test_x)

accuracy = accuracy_score(test_y, predict) * 100
print("A acurácia foi de %.2f%%" % accuracy)