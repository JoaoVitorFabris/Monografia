import numpy as np
import pandas as pd
from keras.layers import Dense, Dropout, Activation, Input
from keras.models import Model

baseInput = pd.read_csv('Inputs_VOPQ.csv')
baseOutput = pd.read_csv('Output_DL2.csv')


# Apagar alguns campos
    
baseInput = baseInput.drop('V1',axis = 1)    # Apaga a coluna
baseInput = baseInput.drop('V2',axis = 1) 
baseInput = baseInput.drop('V3',axis = 1) 
baseInput = baseInput.drop('V6',axis = 1)
baseInput = baseInput.drop('V8',axis = 1)  

baseInput = baseInput.drop('O1',axis = 1)    # Apaga a coluna
baseInput = baseInput.drop('O2',axis = 1) 
baseInput = baseInput.drop('O3',axis = 1) 
baseInput = baseInput.drop('O6',axis = 1)
baseInput = baseInput.drop('O8',axis = 1)  

baseInput = baseInput.iloc[:,0:18].values

baseOutput = baseOutput.drop('DL1',axis = 1)
baseOutput = baseOutput.drop('DL2',axis = 1)
baseOutput = baseOutput.drop('DL3',axis = 1)
baseOutput = baseOutput.drop('DL6',axis = 1)
baseOutput = baseOutput.drop('DL8',axis = 1)

  
# base = base.Output.drop('Developer',axis = 1)

# base = base.dropna(axis = 0)                # Apaga todas as linhas que tiverem 'Nan'
# base = base.loc[base['NA_Sales'] > 1]       # Pega somente valores maiores que 1 na coluna 'NA_Sales'
# base = base.loc[base['EU_Sales'] > 1]       # Pega somente valores maiores que 1 na coluna 'EU_Sales'

# base['Name'].value_counts()
# nome_jogos = base.Name
# base = base.drop('Name',axis = 1)

# previsores = base.iloc[:,[0,1,2,3,7,8,9,10,11]].values
# venda_na = base.iloc[:,4].values
# venda_eu = base.iloc[:,5].values
# venda_jp = base.iloc[:,6].values

DL4 = baseOutput.iloc[:,0].values
DL5 = baseOutput.iloc[:,1].values
DL7 = baseOutput.iloc[:,2].values
DL9 = baseOutput.iloc[:,3].values
DL10 = baseOutput.iloc[:,4].values
DL11 = baseOutput.iloc[:,5].values
DL12 = baseOutput.iloc[:,6].values
DL13 = baseOutput.iloc[:,7].values
DL14 = baseOutput.iloc[:,8].values

#Outra maneira alternativa ao SKlearn

camada_entrada = Input(shape=(18,))
camada_oculta1 = Dense(units = 14, activation='sigmoid')(camada_entrada) # (input+output)/2 -> (28+14)/2
camada_oculta2 = Dense(units = 14, activation='sigmoid')(camada_oculta1)

#Nome camada de saída = topologia de rede neural(parâmetros)(camada a qual está ligada)
out4 = Dense(units = 1, activation='linear')(camada_oculta2)
out5 = Dense(units = 1, activation='linear')(camada_oculta2)
out7 = Dense(units = 1, activation='linear')(camada_oculta2)
out9 = Dense(units = 1, activation='linear')(camada_oculta2)
out10 = Dense(units = 1, activation='linear')(camada_oculta2)
out11 = Dense(units = 1, activation='linear')(camada_oculta2)
out12 = Dense(units = 1, activation='linear')(camada_oculta2)
out13 = Dense(units = 1, activation='linear')(camada_oculta2)
out14 = Dense(units = 1, activation='linear')(camada_oculta2)



regressor = Model(inputs = camada_entrada, 
                  outputs = [out4,out5,out7,out9,out10,out11,out12,out13,out14])
regressor.compile(optimizer = 'adam', loss = 'mse')
regressor.fit(baseInput, [DL4,DL5,DL7,DL9,DL10,DL11,DL12,DL13,DL14],
              epochs = 15000, batch_size = 20) #Batch_size = quantidade de testes antes de atualizar os pesos


out4, out5, out7, out9, out10, out11, out12, out13, out14 = regressor.predict(baseInput)
#Predicao = regressor.predict(baseInput)

#Predicao = pd.DataFrame((out4, out5, out7, out9, out10, out11, out12, out13, out14), columns=['out4', 'out5', 'out7', 'out9', 'out10', 'out11', 'out12', 'out13', 'out14'])

Predicao = np.concatenate((out4, out5, out7, out9, out10, out11, out12, out13, out14),axis=1)
#Predicao = pd.DataFrame(Predicao,columns=['out4', 'out5', 'out7', 'out9', 'out10', 'out11', 'out12', 'out13', 'out14'])

IndiceDL = baseOutput
IndicePrevistoPelaRNA = Predicao

# IndiceDL.mean()
# IndicePrevistoPelaRNA.mean()




# # ==================== CRIAR ESTRUTURA DA RNA EM JSON ====================== #

# regressao_json = regressor.to_json() # Cria variável JSON
# with open('Regressao1.json','w') as json_file: json_file.write(regressao_json) # Salvar em disco

# # ==================== SALVAR OS PESOS ===================================== #

# regressor.save_weights('regressor_v1.h5') #RNA - V1




























