# features (0 = não, 1 = sim)
# pelo longo?
# perna curta?
# faz auau?

porco1 = [0, 1, 0]
porco2 = [0, 1, 1]
porco3 = [1, 1, 0]

cachorro1 = [0, 1, 1]
cachorro2 = [1, 0, 1]
cachorro3 = [1, 1, 1]

# 0 = cachorro, 1 = porco
dados = [porco1, porco2, porco3, cachorro1, cachorro2, cachorro3]
classes = [1, 1, 1, 0, 0, 0]

from sklearn.svm import LinearSVC

model = LinearSVC()
model.fit(dados, classes)

animal_misterioso = [1, 1, 1]
print(model.predict([animal_misterioso]))

animal_misterioso1 = [1, 1, 1]
animal_misterioso2 = [1, 1, 0]
animal_misterioso3 = [0, 1, 1]

testes = [animal_misterioso1, animal_misterioso2, animal_misterioso3]
previsoes = model.predict(testes)
print(previsoes)

testes_classes = [0, 1, 1]

from sklearn.metrics import accuracy_score

taxa_de_acerto = accuracy_score(testes_classes, previsoes)
print("Taxa de acerto: ", taxa_de_acerto * 100)