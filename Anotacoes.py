

""" ===================== CRIAR ESTRUTURA DA RNA EM JSON ====================== """


# regressao_json = regressao.to_json() # Cria variável JSON
# with open('Regressao3.json','w') as json_file: json_file.write(regressao_json) # Salvar em disco

""" ===================== SALVAR OS PESOS ===================================== """

# regressor.save_weights('regressor_v3.h5') #RNA - V3



""" ==================== SALVAR OS PESOS ====================================== """

# estimator = KerasRegressor(build_fn=regressor, nb_epoch=100, batch_size=10, verbose=0)



""" ==================== PLOT CURVA DE APRENDIZADO ============================ """


# import matplotlib.pyplot as plt

# x = hist['epoch']
# y = hist['loss']

# # plot
# fig, ax = plt.subplots()

# ax.plot(x, y, linewidth=1.0)

# plt.show()
