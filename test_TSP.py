import pytest
import numpy as np
import tsplib95
import pdb
from TSP import G, TSP
from networkx.linalg.graphmatrix import adjacency_matrix

def test_G():
    assert G()
    
def test_G_sem_vertices():
    g = G()
    assert np.allclose(g.adjacencias, np.array([]))
    
def test_G_2_vertices():
    g = G(v = 2)
    assert np.allclose(g.adjacencias, np.zeros((2, 2)))

def test_set_matriz_G():
    matriz_distancias = np.ones((5,5))
    for i in range(5):
        matriz_distancias[i][i] = 0
    g = G(grafo = matriz_distancias)
    assert np.allclose(g.adjacencias, matriz_distancias)

def test_set_dicionario_G():
    dicionario_distancias = {
        0: [(1, 1), (2, 1), (3, 1), (4, 1)],
        1: [(0, 1), (2, 1), (3, 1), (4, 1)],
        2: [(0, 1), (1, 1), (3, 1), (4, 1)],
        3: [(0, 1), (1, 1), (2, 1), (4, 1)],
        4: [(0, 1), (1, 1), (2, 1), (3, 1)],
    }
    matriz_distancias = np.ones((5,5))
    for i in range(5):
        matriz_distancias[i][i] = 0
    g = G(grafo = dicionario_distancias)
    assert np.allclose(g.adjacencias, matriz_distancias)

def test_tsplib95_G():
    grafo = tsplib95.load('tsp_files/si535.tsp')
    matriz_distancias = adjacency_matrix(grafo.get_graph()).toarray()
    g = G(grafo = grafo)
    assert np.allclose(g.adjacencias, matriz_distancias)

def test_excecao_grafo_invalido():
    with pytest.raises(Exception) as excinfo:
        g = G(grafo = 13)
    assert 'Tipo de grafo Inválido. O grafo deve ser um array numpy, um dicionario, um grafo TSPLIB95 ou None' in excinfo.value.args

def test_vertices_G():
    g = G(grafo = np.array([[0, 1, 5, 4], [1, 0, 2, 6], [5, 2, 0, 3], [4, 6, 3, 0]]))
    assert np.allclose(g.vertices, np.arange(4))
    
def test_TSP():
    assert TSP()

def test_TSP_two_vertices():
    g = G(grafo = np.array([[0, 2], [2, 0]]))
    tsp = TSP(g = g, origem = 0)
    path, cost, steps = tsp.solve()
    assert np.allclose(path, np.array([0, 1, 0])) and cost == 4 and steps == 1

def test_TSP__brute_force_algorithm_four_vertices():
    g = G(grafo = np.array([[0, 1, 5, 4], [1, 0, 2, 6], [5, 2, 0, 3], [4, 6, 3, 0]]))
    tsp = TSP(g = g, origem = 0)
    path, cost, steps = tsp._TSP__brute_force_algorithm()
    assert np.allclose(path, np.array([0, 1, 2, 3, 0])) and cost == 10 and steps == 6

def test_TSP__nearest_neighbour_heuristic_four_vertices():
    g = G(grafo = np.array([[0, 1, 5, 4], [1, 0, 2, 6], [5, 2, 0, 3], [4, 6, 3, 0]]))
    tsp = TSP(g = g, origem = 0)
    path, cost, steps = tsp._TSP__nearest_neighbour_heuristic()
    assert np.allclose(path, np.array([0, 1, 2, 3, 0])) and cost == 10 and steps == 3