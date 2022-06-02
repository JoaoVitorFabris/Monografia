"""
Com separação treinamento-teste (Holdout)

"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd

baseInput = pd.read_csv('Inputs_VOPQ.csv')
baseOutput = pd.read_csv('Output_DL2.csv')


DelColIn = ['V1', 'V2', 'V3', 'V6', 'V8', 'O1','O2', 'O3','O6', 'O8'] # Colunas a serem apagadas
DelColOut = ['DL1','DL2','DL3','DL6','DL8']                           # Colunas a serem apagadas


baseInput = baseInput.loc[:, ~baseInput.columns.isin(DelColIn)]
baseOutput = baseOutput.loc[:, ~baseOutput.columns.isin(DelColOut)]


from sklearn.model_selection import train_test_split

#20% dos dados para teste e 80% para treinamento
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(baseInput, baseOutput, test_size=0.20)


import keras
from keras.models import Sequential
from keras.layers import Dense

regressor = Sequential()
regressor.add(Dense(units = 14, activation = 'linear', input_dim = 18))
regressor.add(Dense(units = 14, activation = 'linear'))
regressor.add(Dense(units = 14, activation = 'linear'))
regressor.add(Dense(units = 9, activation = 'linear'))

regressor.compile(optimizer = 'adam', loss = 'mean_absolute_error')

regressor.summary()
#history = regressor.fit(previsores_treinamento, previsores_teste, batch_size=30, epochs=500)
history = regressor.fit(previsores_treinamento, classe_treinamento, batch_size=30, epochs=500)
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch           # Armazena os valores dos erros no treinamento
hist.tail()

print(history.params)
print(history.history.keys())
previsoes = regressor.predict(previsores_teste)

import matplotlib.pyplot as plt

x = hist['epoch']
y = hist['loss']

# plot
fig, ax = plt.subplots()

ax.plot(x, y, linewidth=1.0)

plt.show()

resultado = regressor.evaluate(previsores_teste, classe_teste)

print(resultado)
















