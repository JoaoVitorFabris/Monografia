"""

Com Validação Cruzada

"""

import pandas as pd

baseInput = pd.read_csv('Inputs_VOPQ.csv')
baseOutput = pd.read_csv('Output_DL2.csv')


DelColIn = ['V1', 'V2', 'V3', 'V6', 'V8', 'O1','O2', 'O3','O6', 'O8'] # Colunas a serem apagadas
DelColOut = ['DL1','DL2','DL3','DL6','DL8']                           # Colunas a serem apagadas


baseInput = baseInput.loc[:, ~baseInput.columns.isin(DelColIn)]
baseOutput = baseOutput.loc[:, ~baseOutput.columns.isin(DelColOut)]

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import cross_validate
import time

inicio = time.time()

def criarRede():
    regressor = Sequential() # classificador = nome da rede neural 
    regressor.add(Dense(units = 14, activation = 'linear', kernel_initializer = 'random_uniform', input_dim = 18))
    regressor.add(Dropout(0.2)) # Zera 20% dos neurônios aleatoriamente
    regressor.add(Dense(units = 14, activation = 'linear', kernel_initializer = 'random_uniform')) # Input dim coloca somente na primeira camada
    regressor.add(Dropout(0.2)) # Zera 20% dos neurônios aleatoriamente
    regressor.add(Dense(units = 9, activation = 'linear'))
    #otimizador = tf.keras.optimizers.Adam(lr = 0.001, decay = 0.0001, clipvalue = 0.5) #Falta importar TensorFlow
    regressor.compile(optimizer = 'adam', loss = 'mean_absolute_error')
    
    return regressor

regressao = KerasRegressor(model = criarRede, epochs = 10, batch_size = 50)


results = cross_validate(estimator = regressao, X = baseInput, y = baseOutput, cv=10)
a = results['test_score']
b = a.mean()
c = a.std()

previsao = regressao.predict(baseOutput)


print(fim - inicio)


# # ==================== CRIAR ESTRUTURA DA RNA EM JSON ====================== #

# regressao_json = regressao.to_json() # Cria variável JSON
# with open('Regressao3.json','w') as json_file: json_file.write(regressao_json) # Salvar em disco

# # ==================== SALVAR OS PESOS ===================================== #

# regressor.save_weights('regressor_v3.h5') #RNA - V3



# ==================== SALVAR OS PESOS ===================================== #

# estimator = KerasRegressor(build_fn=regressor, nb_epoch=100, batch_size=10, verbose=0)
