import logging
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from icecream import ic

class Problem:
    _graph: nx.Graph
    _alpha: float
    _beta: float

    def __init__(
        self,
        num_cities: int,
        *,
        alpha: float = 1.0,
        beta: float = 1.0,
        density: float = 0.5,
        seed: int = 42,
    ):
        rng = np.random.default_rng(seed)
        self._alpha = alpha
        self._beta = beta
        cities = rng.random(size=(num_cities, 2))
        cities[0, 0] = cities[0, 1] = 0.5

        self._graph = nx.Graph()
        self._graph.add_node(0, pos=(cities[0, 0], cities[0, 1]), gold=0)
        for c in range(1, num_cities):
            self._graph.add_node(c, pos=(cities[c, 0], cities[c, 1]), gold=(1 + 999 * rng.random()))

        tmp = cities[:, np.newaxis, :] - cities[np.newaxis, :, :]
        d = np.sqrt(np.sum(np.square(tmp), axis=-1))
        for c1, c2 in combinations(range(num_cities), 2):
            if rng.random() < density or c2 == c1 + 1:
                self._graph.add_edge(c1, c2, dist=d[c1, c2])

        assert nx.is_connected(self._graph)

    @property
    def graph(self) -> nx.Graph:
        return nx.Graph(self._graph)

    @property
    def alpha(self):
        return self._alpha

    @property
    def beta(self):
        return self._beta

    def cost(self, path, weight):
        dist = nx.path_weight(self._graph, path, weight='dist')
        return dist + (self._alpha * dist * weight) ** self._beta

    def baseline(self):
        cost = 0
        for dest, path in nx.single_source_dijkstra_path(
            self._graph, source=0, weight='weight'
        ).items():
            if dest == 0:
                continue
            logging.debug(
                f"dummy_solution: go to {dest} ({' > '.join(str(n) for n in path)}) -- cost: {self.cost(path, 0):.2f}"
            )
            logging.debug(f"dummy_solution: grab {self._graph.nodes[dest]['gold']:.2f}kg of gold")
            logging.debug(
                f"dummy_solution: return to 0 ({' > '.join(str(n) for n in reversed(path))}) -- cost: {self.cost(path, self._graph.nodes[dest]['gold']):.2f}"
            )
            cost += self.cost(path, 0) + self.cost(path, self._graph.nodes[dest]['gold'])
        logging.info(f"dummy_solution: total cost: {cost:.2f}")
        return cost

    def plot(self):
        plt.figure(figsize=(10, 10))
        pos = nx.get_node_attributes(self._graph, 'pos')
        size = [100] + [self._graph.nodes[n]['gold'] for n in range(1, len(self._graph))]
        color = ['red'] + ['lightblue'] * (len(self._graph) - 1)
        return nx.draw(self._graph, pos, with_labels=True, node_color=color, node_size=size)
    
    
    from src.genetic_solver import GeneticSolver
    def solution(self):
        """
        Risolve il problema utilizzando un Algoritmo Genetico con decodifica intelligente.
        """
        import logging
        
        # Configurazione parametri GA
        # Sentiti libero di aumentarli per risultati migliori (es. pop=200, gen=500)
        POPULATION_SIZE = 10
        GENERATIONS = 20
        MUTATION_RATE = 0.3
        
        logging.info(f"Starting Genetic Algorithm (Pop: {POPULATION_SIZE}, Gen: {GENERATIONS})...")
        # Istanzia ed esegue il solver
        solver = self.GeneticSolver(
            problem=self, 
            pop_size=POPULATION_SIZE, 
            generations=GENERATIONS, 
            mutation_rate=MUTATION_RATE
        )

        best_individual = solver.evolve()
        
        # Logging del risultato finale (simile alla baseline)
        path = best_individual.phenotype
        cost = best_individual.fitness
        
        logging.info(f"Solution found with cost: {cost:.2f}")
        logging.info(f"Path length: {len(path)} steps")
        logging.debug(f"Full path: {path}") # De-commentare se serve vedere il percorso
        return cost
    
    def compare(self):
        baseline_cost = self.baseline()
        solution_cost = self.solution()
        improvement = (baseline_cost - solution_cost) / baseline_cost * 100
        return (improvement, solution_cost, baseline_cost)

if __name__ == "__main__":
    out = open("results.txt", "w")
    #possible values num_cities: 100, 1_000, density: 0.2, 1, alpha: 1, 2, beta: 1, 2
    for num_cities in [100]:
        for density in [0.2, 1]:
            for beta in [1, 2]:
                for alpha in [1, 2]:
                    print(f"Running Problem with {num_cities} cities, density={density}, alpha={alpha}, beta={beta}")
                    (improvment, sol_cost, base_cost) = Problem(100, density=density, alpha=alpha, beta=beta).compare()
                    out.write(f"Density: {density}, Alpha: {alpha}, Beta: {beta} => Improvement: {improvment:.2f}%, Solution Cost: {sol_cost:.2f}, Baseline Cost: {base_cost:.2f}\n")
    out.close()
