import pandas as pd

def saveXLSX(a,b):
    diretorio = 'Resultados\\teste.xlsx'

    Mae = pd.read_excel(diretorio, sheet_name='MAE', index_col=0)
    Params = pd.read_excel(diretorio, sheet_name='Params', index_col=0)
    
    Mae.loc[len(Mae)] = a
    Params.loc[len(Params)] = b

    with pd.ExcelWriter(diretorio) as writer:
        Mae.to_excel(writer, sheet_name='MAE')
        Params.to_excel(writer, sheet_name='Params')


def norm(x, stats):
    normVal = (x - stats['mean']) / stats['std']
    flag_norm = True
    return(normVal,flag_norm)

def norm2(x, stats):
    Vmax = stats['max']
    Vmin = stats['min']
    
    normVal = (x - Vmin) / (Vmax-Vmin)
    flag_norm = True
    return(normVal,flag_norm)

def disnorm(x, stats):
    x = pd.DataFrame(x)
    aaa = len(x.columns())
    mean = stats['mean']
    std = stats['std']
    disnormVal = stats['mean'] + stats['std']*x
    
    return(disnormVal)

# Não funciona

def disnorm2(x, stats):
    # Corrigir para percorrer cada valor e muliplicar
    df_cstats = stats.copy()
    df_cstats_std = stats['std']
    df_cstats_mean = stats['mean']
    a = len(df_cstats.index.values)
    b = df_cstats.columns
    c = 1
    disnormVal = []
    for ind in range(len(df_cstats.index)):
        for col in range(len(x.columns)):
            disnormVal[ind][col] = x[ind][col]*df_cstats_std[ind]+ df_cstats_mean[ind]
    return(disnormVal)
