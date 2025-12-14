import logging
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import time
from icecream import ic
from src.solver_framework import problem_solver
    
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
        return self._graph

    @property
    def alpha(self):
        return self._alpha

    @property
    def beta(self):
        return self._beta

    def cost(self, path, weight):
        """
        Cost to traverse a path carrying constant weight
        
        :param path: nodes in the path
        :param weight: constant weight carried
        """
        return sum(
            self.adj_cost(path[i], path[i+1], weight)
            for i in range(len(path) - 1)
        )
    
    def adj_cost(self, src, dest, weight):
        """
        Cost to go from src to dest (adjacent nodes) carrying weight
        
        :param src: starting city
        :param dest: destination city
        :param weight: weight carried
        """
        dist = self._graph[src][dest]['dist']
        return dist + (self._alpha * dist * weight) ** self._beta
    
    def path_cost(self, path: list[tuple[int, float]]) -> float:
        """
        Calculates the total cost of traversing the given path.
        Iterates through edges (u -> v) to ensure consistency with the optimizer logic.
        
        :param path: Sequence of (city, gold to pick up at city)
                        Example: [(0, 0), (20, 1000), (0, 0)]
        :type path: list[tuple[int, float]]
        """
        if path[0][0] != 0:
                path = [(0, 0.0)] + path

        total_cost = 0.0
        current_weight = 0.0

        # Iterate through each edge in the path
        for i in range(len(path) - 1):
            u, gold_u = path[i]
            v, gold_v = path[i+1]

            # Discharge logic: if we return to the depot (node 0), reset weight
            if u == 0:
                current_weight = 0.0

            # Pick up gold at node u
            current_weight += gold_u

            # Compute cost to go from u to v with current weight
            total_cost += self.adj_cost(u, v, current_weight)

        return total_cost

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
    
    def solution(self):
        """
        Solve the problem using multiple solvers and select the best solution.
        """
        
        return problem_solver(self)

    
    def compare(self):
        baseline_cost = self.baseline()
        _, solution_cost = self.solution()
        improvement = (baseline_cost - solution_cost) / baseline_cost * 100
        return (improvement, solution_cost, baseline_cost)
    




if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    out = open("results.txt", "w")
    # Possible values: num_cities: 100, 1_000; density: 0.2, 1; alpha: 1, 2; beta: 1, 2
    for num_cities in [100]:
        for density in [0.2, 1]:
            for beta in [0.0, 0.5, 1, 2]:
                for alpha in [0, 1, 2]:
                    print(f"Running Problem with {num_cities} cities, density={density}, alpha={alpha}, beta={beta}")
                    start_time = time.time()
                    (improvment, sol_cost, base_cost) = Problem(100, density=density, alpha=alpha, beta=beta).compare()
                    elapsed_time = time.time() - start_time
                    out.write(f"Density: {density}, Alpha: {alpha}, Beta: {beta} => Improvement: {improvment:.2f}%, Solution Cost: {sol_cost:.2f}, Baseline Cost: {base_cost:.2f}, Time: {elapsed_time:.2f}s\n")
    out.close()
