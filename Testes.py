import pandas as pd
from keras.layers import Dense, Dropout, Activation, Input
from keras.models import Model

nomestring = "output"

nb = 14
camada_entrada = 1

camada_entrada = Input(shape=(61,))
camada_oculta1 = Dense(units = 32, activation='sigmoid')(camada_entrada) # (input+output)/2 -> (61+3)/2
camada_oculta2 = Dense(units = 32, activation='sigmoid')(camada_oculta1)

for i in range(nb):
     print(i)
     indexstring = str(i)
     var = nomestring + indexstring
     exec("%s = %d" % (var,i))
     #var = Dense(units = 1, activation='linear')(camada_oculta2)

#print(output8)

#Dense(units = 1, activation='linear')(camada_oculta2)

