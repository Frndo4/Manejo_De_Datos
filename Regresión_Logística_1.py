import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split #Seleccionamos un módulo
#Validación de la BD usando Machine Learning
from sklearn.linear_model import LogisticRegression
from sklearn import metrics #Metrica o predicción del modelo
import matplotlib.pyplot as plt
#Cargar BD
url=r'C:\Users\fer-j\Documents\Python\Inteligencia_Artificial\Manejo_de_Datos\default.csv'
datos=pd.read_csv(url, sep=',')
print(datos.head(100))

print(datos[0:6])
print(len(datos))

#Estimando modelo de regresión logística

#Asignar variables predictoras

X=datos[['student','balance','income']]
y=datos[['default']]

print(X.head())
print(y.head())

X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.3, random_state=0)

#Instanciar el modelo}
regresión_logística=LogisticRegression()

#Ajustar el modelo usando el training
regresión_logística.fit(X_train, y_train)

#Usando el modelo para hacer predicciones del testing
y_pred = regresión_logística.predict(X_test)
print(y_pred)

#Diagnóstico del modelo
cnf_matriz=metrics.confusion_matrix(y_test, y_pred)
print(cnf_matriz)

print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))