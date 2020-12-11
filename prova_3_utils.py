import numpy as np
import tsplib95
from tabulate import tabulate

def gerar_adjacencia_simetrica_aleatoria(N):
    a = np.random.randint(1, 100, (N, N))
    for i in range(N):
        a[i][i] = 0
    return np.tril(a) + np.tril(a, -1).T

def repr_matriz(matriz):
    header = [i for i in np.arange(matriz.shape[0])]
    return tabulate(matriz, headers = header, showindex=True, tablefmt = 'fancy_grid')

def carregar_grafo_tsp(filepath):
    return tsplib95.load(filepath)