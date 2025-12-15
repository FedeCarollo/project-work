import logging
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
    
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

    def cost(self, path, weight):
        dist = nx.path_weight(self._graph, path, weight='dist')
        return dist + (self._alpha * dist * weight) ** self._beta

    def baseline(self):
        total_cost = 0
        for dest, path in nx.single_source_dijkstra_path(
            self._graph, source=0, weight='dist'
        ).items():
            cost = 0
            for c1, c2 in zip(path, path[1:]):
                cost += self.cost([c1, c2], 0)
                cost += self.cost([c1, c2], self._graph.nodes[dest]['gold'])
            logging.debug(
                f"dummy_solution: go to {dest} ({' > '.join(str(n) for n in path)} ({cost})"
            )
            total_cost += cost
        return total_cost

    def plot(self):
        plt.figure(figsize=(10, 10))
        pos = nx.get_node_attributes(self._graph, 'pos')
        size = [100] + [self._graph.nodes[n]['gold'] for n in range(1, len(self._graph))]
        color = ['red'] + ['lightblue'] * (len(self._graph) - 1)
        return nx.draw(self._graph, pos, with_labels=True, node_color=color, node_size=size)
    