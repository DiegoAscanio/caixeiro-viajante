import numpy as np
import pdb
from tsplib95.models import StandardProblem
from networkx.linalg.graphmatrix import adjacency_matrix
from sympy.utilities.iterables import multiset_permutations

class G:
    '''
    Definicao da classe Grafo, para armazenar os grafos para o TSP
    A classe aceita grafos de até 4 tipos:
       1. Array (numpy)
       2. Dicionario (python)
       3. Grafo TSPLIB95 (arquivo tsp)
       4. None
    Os vértices são dados por números inteiros contíguos
    e a representação do grafo é dada por matriz de adjacências
    onde a aresta entre um par de vértices u e v é representada
    na matriz de adjacências self.adjacencias[u][v] pela distância
    entre os vértices, ou seja:
        self.adjacencias[u][v] = distancia
    Também entende-se implicitamente o grafo como completo e que portanto
    a matriz de adjacências é simétrica e que nenhum vértice possui ares-
    tas para si mesmo, ou seja:
        self.adjacencias[u][v] == self.adjacencias[v][u] (∀ u, v ∈ G),
        self.adjacencias[u][u] == 0 (∀ u ∈ G)
    '''
    def __init__(self, grafo = None, v = 0):
        if grafo is not None:
            if type(grafo) == np.ndarray:
                self.adjacencias = grafo
            elif type(grafo) == dict:
                v = len(grafo)
                self.adjacencias = np.zeros((v, v))
                for origem in grafo:
                    for destino, distancia in grafo[origem]:
                        self.adjacencias[origem][destino] = distancia
            elif type(grafo) == StandardProblem:
                self.adjacencias = adjacency_matrix(grafo.get_graph()).toarray()
            else:
                raise Exception('Tipo de grafo Inválido. O grafo deve ser um array '+
                                'numpy, um dicionario, um grafo TSPLIB95 ou None')
        else:
            self.adjacencias = np.zeros((v, v))
    @property
    def vertices(self):
        return np.arange(self.adjacencias.shape[0])

class TSP:
    def __init__(self, g = None, origem = 0):
        self.g = g
        self.origem = origem
    def __brute_force_algorithm(self):
        '''
        Definição do método força bruta:
        
        Dado um grafo G com n vértices e um vértice de origem O, avalia todas
        as (n - 1)! permutações possíveis para encontrar o caminho mínimo para
        a viagem do caixeiro viajante.
        '''
        
        # cidades_destinos são todos os (n - 1) vértices do grafo G
        # à exceção da origem
        cidades_destinos = np.concatenate((self.g.vertices[0:self.origem],
                                           self.g.vertices[self.origem + 1:]))
        # multiset_permutations são as (n - 1)! permutações de cidades destinos
        caminhos = multiset_permutations(cidades_destinos)
        caminho_minimo = np.zeros(len(self.g.vertices + 1))
        custo_caminho_minimo = np.inf
        steps = 0
        
        # itera em cada caminho da permutação
        for caminho in caminhos:
            # cada caminho começa na origem
            origem = self.origem
            custo_caminho = 0
            for destino in caminho:
                # passa por todas as cidades da permutação
                custo_caminho += self.g.adjacencias[origem][destino]
                origem = destino
            # termina no destino
            destino = self.origem
            custo_caminho += self.g.adjacencias[origem][destino]
            
            if custo_caminho < custo_caminho_minimo:
                # se a permutacao atual for a de menor caminho
                # atualiza o menor caminho
                custo_caminho_minimo = custo_caminho
                caminho_minimo = np.concatenate((np.array([self.origem]), caminho, np.array([self.origem])))
            steps += 1
        
        return caminho_minimo, custo_caminho_minimo, steps
    
    def __nearest_neighbour_heuristic(self):
        '''
        Definição da abordagem heurística do vizinho mais próximo:
        1. Defina u como o vértice de origem do caixeiro viajante
        2. Enquanto Houverem Vértices a Serem Visitados:
            2.1 Visite o vértice v mais próximo adjacente a u
            2.2 Adicione v aos vértices visitados
            2.2 Atualize u para o ultimo vértice v visitado
            
        Essa heurística não é ótima, segundo JONHSON e MCGEOCH(1995)
        a heurística retorna em média caminhos 25% maiores do que
        o caminho mínimo, mas, é uma heurística fácil de ser implemen-
        tada e que retorna um caminho candidato em custo O(n^2).
        Por isso, é a heurística escolhida para resolver o problema
        do caixeiro viajante
        '''
        visitados = [self.origem] # visita o vértice de origem
        custo_caminho = 0
        steps = 0
        
        # a condição de parada é que a lista ordenada de vértices
        # visitados seja igual a lista de vértices do grafo G, que
        # por padrão, é definida ordenda
        while sorted(visitados) != list(self.g.vertices):
            # quando uso indexacao negativa em python quer dizer que desejo
            # acessar o ultimo elemento de uma determinada lista
            # Neste caso, u = visitados[-1] executa a instrução 2.2
            # (Atualize u para o ultimo vértice v visitado)
            u = visitados[-1]
            distancia_vizinho_mais_proximo = np.inf
            for v in list(self.g.vertices)[0:u] + list(self.g.vertices)[u + 1:]:
                # Compara os vertices v adjacentes a u para escolher o vertice adjacente mais proximo
                # que não tenha sido visitado para compor o caminho da solução
                if self.g.adjacencias[u][v] < distancia_vizinho_mais_proximo and v not in visitados:
                    distancia_vizinho_mais_proximo = self.g.adjacencias[u][v]
                    vizinho_mais_proximo = v
                steps += 1
            # adiciona o vizinho v mais proximo de u na lista de vertices
            # visitados e adiciona a distancia de v a u ao custo do caminho
            # solucao
            visitados.append(vizinho_mais_proximo)
            custo_caminho += self.g.adjacencias[u][vizinho_mais_proximo]
        
        # finaliza o caminho, fechando o ciclo do passeio, retornando para a origem
        visitados.append(self.origem)
        custo_caminho += self.g.adjacencias[vizinho_mais_proximo][self.origem]
        
        return visitados, custo_caminho, steps
    
    def solve(self, heuristic = True):
        # resolve o caixeiro viajante de acordo com o método especificado:
        # pela heurística do vizinho mais próximo se heuristic for True
        # ou por força bruta se heuristic for False
        if heuristic:
            return self.__nearest_neighbour_heuristic()
        else:
            return self.__brute_force_algorithm()