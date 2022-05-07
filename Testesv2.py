import pandas as pd

baseInput = pd.read_csv('Inputs_VOPQ.csv')
baseOutput = pd.read_csv('Output_DL2.csv')


DelColIn = ['V1', 'V2', 'V3', 'V6', 'V8', 'O1','O2', 'O3','O6', 'O8'] # Colunas a serem apagadas
DelColOut = ['DL1','DL2','DL3','DL6','DL8']                           # Colunas a serem apagadas


baseInput = baseInput.loc[:, ~baseInput.columns.isin(DelColIn)]
baseOutput = baseOutput.loc[:, ~baseOutput.columns.isin(DelColOut)]

import keras
from keras.models import Sequential
from keras.layers import Dense

regressor = Sequential()
regressor.add(Dense(units = 14, activation = 'sigmoid', input_dim = 18))
regressor.add(Dense(units = 14, activation = 'sigmoid'))
regressor.add(Dense(units = 14, activation = 'sigmoid'))
regressor.add(Dense(units = 9, activation = 'linear'))

regressor.compile(optimizer = 'adam', loss = 'mean_absolute_error')



regressor.summary()
history = regressor.fit(baseInput, baseOutput, batch_size=30, epochs=5000)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch           # Armazena os valores dos erros no treinamento
hist.tail()

previsoes = regressor.predict(baseInput)

resultado = regressor.evaluate(baseInput, baseOutput)


# ==================== CRIAR ESTRUTURA DA RNA EM JSON ====================== #

regressao_json = regressor.to_json() # Cria variável JSON
with open('Regressao2.json','w') as json_file: json_file.write(regressao_json) # Salvar em disco

# ==================== SALVAR OS PESOS ===================================== #

regressor.save_weights('regressor_v2.h5') #RNA - V1

# estimator = KerasRegressor(build_fn=regressor, nb_epoch=100, batch_size=10, verbose=0)

# from keras.wrappers.scikit_learn import KerasRegressor
# from sklearn.model_selection import cross_val_score

# def criarRede():
#     regressor = Sequential() # classificador = nome da rede neural 
#     regressor.add(Dense(units = 14, activation = 'linear', kernel_initializer = 'random_uniform', input_dim = 18))
#     # regressor.add(Dropout(0.2))
#     regressor.add(Dense(units = 14, activation = 'linear', kernel_initializer = 'random_uniform')) # Input dim coloca somente na primeira camada
#     # regressor.add(Dropout(0.2))
#     regressor.add(Dense(units = 9, activation = 'linear'))
#     #otimizador = tf.keras.optimizers.Adam(lr = 0.001, decay = 0.0001, clipvalue = 0.5) #Falta importar TensorFlow
#     regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
    
#     return regressor

# regressor = KerasRegressor(build_fn = criarRede, epochs = 1000, batch_size = 10)

# results = cross_val_score(estimator = regressor, X = baseInput, y = baseOutput, cv=10, scoring='accuracy')



# import matplotlib.pyplot as plt

# x = hist['epoch']
# y = hist['loss']

# # plot
# fig, ax = plt.subplots()

# ax.plot(x, y, linewidth=1.0)

# plt.show()
 
