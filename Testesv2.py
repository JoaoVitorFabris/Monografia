"""
Com separação treinamento-teste (Holdout)

"""

import pandas as pd
import myfuncs as mf  # Arquivo 'myFuncs.py'
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense


flag_norm = True

for i in range(1):

    baseInput = pd.read_csv('Inputs_VOPQ_2.csv')
    baseOutput = pd.read_csv('Output_DL2_2.csv')

    DelColIn = ['V1', 'V2', 'V3', 'V6', 'V8', 'O1', 'O2',
                'O3', 'O6', 'O8']  # Colunas a serem apagadas
    # Colunas a serem apagadas
    DelColOut = ['DL1', 'DL2', 'DL3', 'DL6', 'DL8']

    baseInput = baseInput.loc[:, ~baseInput.columns.isin(DelColIn)]
    baseOutput = baseOutput.loc[:, ~baseOutput.columns.isin(DelColOut)]
      #20% dos dados para teste e 80% para treinamento
    test_size = 0.2

    if flag_norm:
         classe_stats = baseOutput.describe().transpose()
         previsores_stats = baseInput.describe().transpose()
         baseOuputNorm, flag_norm = mf.norm2(baseOutput, classe_stats)
         baseInputNorm, flag_norm = mf.norm2(baseInput, previsores_stats)
         previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(
             baseInputNorm, baseOuputNorm, test_size=test_size)
    else:
        previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(baseInput, baseOutput, test_size=test_size)
        
    dim_input = str(list(baseInput.shape))
    dim_output = str(list(baseOutput.shape))
    
    input_dim = len(baseInput.columns)
    output_dim = len(baseOutput.columns)
    
    nOculta1 = round((input_dim+output_dim)/2)
    nOculta2 = round((input_dim+output_dim)/2)
    
    activationO1 = 'sigmoid'
    activationO2 = 'sigmoid'
    activationOut = 'linear'
    
    optimizer = 'adam'
    loss = 'mae'
    
    batch_size = 20
    epochs = 100
    
    regressor = Sequential()
    
    regressor.add(Dense(units=nOculta1, activation=activationO1, input_dim=input_dim))
    regressor.add(Dense(units=nOculta2, activation=activationO2))
    regressor.add(Dense(units=output_dim, activation=activationOut))
    
    regressor.compile(optimizer=optimizer, loss=loss)
    
    #regressor.summary()
    
    history = regressor.fit(previsores_treinamento, classe_treinamento, batch_size=batch_size, epochs=epochs)
    #hist = pd.DataFrame(history.history)
    #hist['epoch'] = history.epoch           # Armazena os valores dos erros no treinamento
    #hist.tail()
    
    print(history.params)
    print(history.history.keys())
    previsoes = regressor.predict(previsores_teste)
    
   
    resultado = regressor.evaluate(previsores_teste, classe_teste)
    
    a = abs(previsoes - classe_teste)
    a_mean = a.mean()
    a_std = a_mean.std()
    
    a_mean = a_mean.tolist()
    a_mean.extend([resultado, a_std])
    flag_norm2 = str(flag_norm)
    
    b = [dim_input, dim_output, flag_norm2, nOculta1, nOculta2, activationO1, activationO2, activationOut, 
         optimizer, loss, batch_size, epochs]
    
    #b = a.describe().transpose() # Só funciona se retirar o "tolist()"
    
    # cc = mf.disnorm(previsoes,classe_stats)
    
    mf.saveXLSX(a_mean, b)
