import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

def visualizar(real, prediccion):
    plt.plot(real[0:len(prediccion)],color='red', label='Precio máximo real de la acción')
    plt.plot(prediccion, color='blue', label='Predicción de la acción')
    plt.xlabel('Tiempo')
    plt.ylabel('Precio de la acción')
    plt.legend()
    plt.show()

""" Lectura de los datos, relacionando los valores con su fecha. """
dataset = pd.read_csv('amazonStockHistory.csv', index_col='Date', parse_dates=['Date'])

""" Solo vamos a basarnos en el Precio más alto de la acción (columna 'High') """
trainingSet = dataset['2019':].iloc[:, [False, False, False, True, False]]
testSet = dataset[:'2020'].iloc[:, [False, False, False, True, False]]

""" Normalización del set de entrenamiento (valores entre 0 y 1). """
sc = MinMaxScaler(feature_range=(0,1))
trainingSetScaled = sc.fit_transform(trainingSet)

""" Vamos a entrenar a la Red proporcionando 100 datos de entrada y 1 de salida en cada iteración """
timeSteps = 100
xTrain = []
yTrain = []

""" xTrain = lista de conjuntos de 100 datos.
    yTrain = lista de valores """
for i in range(0, len(trainingSetScaled)-timeSteps):
    xTrain.append(trainingSetScaled[i:i+timeSteps, 0])
    yTrain.append(trainingSetScaled[i+timeSteps,0])

""" Preferiblemente usar numpy ya que:
    1. Deberemos transformar xTrain (actualmente de dos dimensiones) a tres dimensiones.
    2. Los programas que usan Numpy generalmente son más rápidos (sobretodo en IA).
"""
xTrain, yTrain = np.array(xTrain), np.array(yTrain)

""" Hay que añadir una dimensión a xTrain, nos lo pide la libreria Keras """
xTrain = np.reshape(xTrain, (xTrain.shape[0], xTrain.shape[1], 1))

""" Parámetros que deberemos proporcionar a Keras (Sequential()). """
dim_entrada = (xTrain.shape[1],1) # No hace falta la primera.
dim_salida = 1
na = 50

""" units = neuronas de la capa | return_sequences = hay más capas? | input_shape = dimensión entrada | 
    Dropout(%) = Número de neuronas que queremos ignorar en la capa de regularización (normalmente es de un 20%). """
regresor = Sequential() # Inicializa el modelo


""" capa 1 """
regresor.add(LSTM(units=na, input_shape=dim_entrada))

""" capa output """
regresor.add(Dense(units=dim_salida))

regresor.compile(optimizer='rmsprop', loss='mse') # mse = mean_squared_error

""" Encajar Red Neuronal en Set Entrenamiento """
""" epochs = iteraciones para entrenar tu modelo | 
batch_size = numero ejemplos entrenamiento (cuanto más alto, más memoria necesitarás).  """
regresor.fit(xTrain,yTrain,epochs=20, batch_size=32)

""" Normalizar el conjunto de Test y relizamos las mismas operaciones que anteriormente """
auxTest = sc.transform(testSet.values)
xTest = []

for i in range(0, len(auxTest)-timeSteps):
    xTest.append(auxTest[i:i+timeSteps,0])
    
xTest = np.array(xTest)
xTest = np.reshape(xTest, (xTest.shape[0],xTest.shape[1], 1))

""" Realizamos la Predicción """
prediccion = regresor.predict(xTest)

""" Desnormalizamos la Predicción para que se encuentre entre valores normales. """
prediccion = sc.inverse_transform(prediccion)

# Graficar resultados
visualizar(testSet.values,prediccion)


