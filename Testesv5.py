# -*- coding: utf-8 -*-
"""

Tuning dos parâmetros

"""

import pandas as pd
import keras
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Dropout  # Dropout é aula 8!
# Utilizando keras em conjunto com scikit_learn. Keras Classifier = classe
from scikeras.wrappers import KerasRegressor
# Pesquisa em grade para descobrir os melhores parâmetros para a RNA
from sklearn.model_selection import GridSearchCV


baseInput = pd.read_csv('Inputs_VOPQ.csv')
baseOutput = pd.read_csv('Output_DL2.csv')

# Função para criação de rede neural


def criarRede(optimizer, loos, kernel_initializer, activation, neurons):

    regressor = Sequential()  # classificador = nome da rede neural
    regressor.add(Dense(units = neurons, activation = activation, kernel_initializer = kernel_initializer, input_dim = 18))
    regressor.add(Dropout(0.2)) # Zera 20% dos neurônios aleatoriamente
    regressor.add(Dense(units = neurons, activation = activation, kernel_initializer = kernel_initializer)) # Input dim coloca somente na primeira camada
    regressor.add(Dropout(0.2)) # Zera 20% dos neurônios aleatoriamente
    regressor.add(Dense(units = 9, activation = 'linear'))
    #otimizador = tf.keras.optimizers.Adam(lr = 0.001, decay = 0.0001, clipvalue = 0.5) #Falta importar TensorFlow
    regressor.compile(optimizer = optimizer, loss = loss)
      
    
    return regressor

regressao = KerasRegressor(model = criarRede)

parametros = {'batch_size': [10, 30],
              'epochs': [500],
              'optimizer': ['adam','sgd'],
              'loos':['mae','mse'],
              'kernel_initializer': ['random_uniform','normal'],
              'activation': ['relu','linear'],
              'neurons':[16, 8]}
grid_search = GridSearchCV(estimator = regressao,
                           param_grid = parametros,
                           scoring = 'accuracy',
                           cv = 5)
grid_search = grid_search.fit(baseInput,baseOutput)
melhores_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_


