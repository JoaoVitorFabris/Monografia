def saveXLSX(a,b):
    import pandas as pd
    diretorio = 'Resultados\\teste.xlsx'

    Mae = pd.read_excel(diretorio, sheet_name='MAE', index_col=0)
    Params = pd.read_excel(diretorio, sheet_name='Params', index_col=0)
    
    Mae.loc[len(Mae)] = a
    Params.loc[len(Params)] = b

    with pd.ExcelWriter(diretorio) as writer:
        Mae.to_excel(writer, sheet_name='MAE')
        Params.to_excel(writer, sheet_name='Params')


def norm(x, classe_stats):
    normVal = (x - classe_stats['mean']) / classe_stats['std']
    return(normVal)


def disnorm(x, classe_stats):
    # Corrigir para percorrer cada valor e muliplicar
    df_cstats = classe_stats.copy()
    df_cstats_std = df_cstats['std']
    df_cstats_mean = df_cstats['mean']
    a = len(df_cstats.index.values)
    b = df_cstats.columns
    c = 1
    disnormVal = []
    for ind in range(len(df_cstats.index)):
        for col in range(len(x.columns)):
            disnormVal[ind][col] = x[ind][col]*df_cstats_std[ind]+ df_cstats_mean[ind]
    return(disnormVal)
