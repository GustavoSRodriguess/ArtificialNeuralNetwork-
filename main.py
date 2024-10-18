import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

X_and = np.array([
    [1, 1],
    [1, 0],
    [0, 1],
    [0, 0]
])

y_and = np.array([1, 0, 0, 0])

X_or = np.array([
    [1, 1],
    [1, 0],
    [0, 1],
    [0, 0]
])

y_or = np.array([1, 1, 1, 0])

clf_and = Perceptron(random_state=42)
clf_and.fit(X_and, y_and)

clf_or = Perceptron(random_state=42)
clf_or.fit(X_or, y_or)

print("Resultados para AND:")
predictions_and = clf_and.predict(X_and)
print("Predições:", predictions_and)
print("Pesos:", clf_and.coef_[0])
print("Bias:", clf_and.intercept_[0])
print("Acurácia:", accuracy_score(y_and, predictions_and))

print("\nResultados para OR:")
predictions_or = clf_or.predict(X_or)
print("Predições:", predictions_or)
print("Pesos:", clf_or.coef_[0])
print("Bias:", clf_or.intercept_[0])
print("Acurácia:", accuracy_score(y_or, predictions_or))

print("\nTestando casos específicos:")
teste = np.array([[1, 1]])
print("1 AND 1 =", clf_and.predict(teste)[0])
print("1 OR 1 =", clf_or.predict(teste)[0])

#interativo com o user
n1 = int(input('Primeiro numero para comparar (digite 0 ou 1): '))
n2 = int(input('Segundo numero para comparar (digite 0 ou 1): '))

if n1 not in [0, 1] or n2 not in [0, 1]:
    print("Por favor, digite apenas 0 ou 1!")
else:
    entrada_teste = np.array([[n1, n2]]) 
    resultado_and = clf_and.predict(entrada_teste)[0]
    resultado_or = clf_or.predict(entrada_teste)[0]

    print(f"\nResultados:")
    print(f"{n1} AND {n2} = {resultado_and}")
    print(f"{n1} OR {n2} = {resultado_or}")