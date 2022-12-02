import numpy as np                  #Matrices, organización
import pandas as pd                 #Operaciones matemáticas, organización de tablas, arreglos
import matplotlib.pyplot as plt     #Generar gráficos

x=[2, 3, 4, 5, 6, 7, 8, 9, 1, 34]
y=[1, 4, 5, 6, 2, 8, 6, 7, 9, 15]

#Vector X y Y
n=len(x)
x=np.array(x)
y=np.array(y)
print(x, y)

sumx=sum(x)
print('La sumatoria de x = ', sumx)

sumy=sum(y)
print('La sumatoria de y = ', sumy)

sumx2=sum(x*x)
print('La sumatoria de X^2 = ', sumx2)

sumy2=sum(y*y)
print('La sumatoria de Y^2 = ', sumy2)

sumxy=sum(x*y)
print('La sumatoria de XY = ', sumxy)

mediax=sum(x)/n
print('La media de x = ', mediax)

mediay=sum(y)/n
print('La media de y = ', mediay)

#La ecuación de la recta
m=(n*sumxy-sumx*sumy)/(n*sumx2-sumx**2)
print('m= ', m)

b=mediay-m*mediax
print('b= ', b)

print('La ecuación de la recta = ',m,'x + ',b)

#Error
sxy=np.sqrt((sumy2-b*sumy-m*sumxy)/(n-2))
print('Error = ', sxy)

#Coeficiente de correlación
r=(n*sumxy-sumx*sumy)/(np.sqrt(n*sumx2-sumx**2)*((np.sqrt(n*sumy2-sumy**2))))
print('Coeficiente de correlación =' , r)

#Ploteo (Gráfica)
plt.plot(x, y, 'o', label='Datos')   #Datos X y Y
plt.plot(x, m*x +b, label='Ajustes')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Regresión lineal')
plt.grid()
plt.legend(loc=4)
plt.show()