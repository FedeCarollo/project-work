import numpy as np
import networkx as nx
import math
import logging
from typing import List, Tuple
from time import time
from .beta_optimizer import path_optimizer



class ACOSolver:
    """
    Ant Colony Optimization (ACO) implementation.
    This is a Population-Based method
    relying on Stigmergy (indirect communication via Pheromones).

    Strategy:
    1. Construction: Ants build a 'Giant Tour' (visitation order) using Pheromones + Heuristic.
    2. Evaluation: We use the 'Split Algorithm' (Prins) to optimally cut the tour into trips.
    3. Feedback: Good solutions reinforce the trail (Pheromone Deposit); trails decay over time (Evaporation).
    """

    def __init__(self, problem, n_ants=25, n_iterations=60,
                 alpha=1.0, beta=1.0, rho=0.1, q=100.0):
        self.problem = problem
        self.n_ants = n_ants
        self.n_iterations = n_iterations

        # --- Parameter Tuning (Exploration vs Exploitation) ---
        # If the problem is "hard" (Beta >= 2), the cost landscape is very rugged.
        # We rely more on Heuristic info (Visibilty) locally to avoid expensive long jumps,
        # and we increase the population to cover more ground.
        if problem.beta >= 2:
            self.n_ants = 40
            self.n_iterations = 100
            self.aco_alpha = 1.0  # Pheromone importance
            self.aco_beta = 3.0  # Heuristic importance (Greediness)
        else:
            self.aco_alpha = alpha
            self.aco_beta = beta

        self.rho = rho  # Evaporation rate
        self.Q = q  # Pheromone quantity

        # Filter out the depot (0) from the target list
        self.cities = [n for n in problem.graph.nodes if n != 0]
        self.n_cities = len(self.cities)
        self.idx_to_node = {i: n for i, n in enumerate(self.cities)}

        # Pre-compute Dijkstra for O(1) distance lookups.
        # Crucial for sparse graphs where edge(u,v) might not exist.
        self.shortest_paths = dict(nx.all_pairs_dijkstra_path(problem.graph, weight='dist'))
        self.shortest_dists = dict(nx.all_pairs_dijkstra_path_length(problem.graph, weight='dist'))

        # Initialize Pheromone Matrix (The "Memory" of the colony)
        # Small initial value to allow exploration.
        self.pheromone = np.ones((self.n_cities, self.n_cities)) * 0.1

        # Initialize Heuristic Matrix (The "Visibility")
        # Static value: 1 / distance. Ants prefer closer cities.
        self.heuristic = np.zeros((self.n_cities, self.n_cities))
        for i in range(self.n_cities):
            for j in range(self.n_cities):
                if i != j:
                    u, v = self.idx_to_node[i], self.idx_to_node[j]
                    dist = self.shortest_dists[u][v]
                    self.heuristic[i, j] = 1.0 / (dist + 1e-5)

    def solve(self) -> Tuple[List[Tuple[int, float]], float]:
        best_solution_path = None
        best_cost = float('inf')

        # Main Evolution Loop
        for it in range(self.n_iterations):
            iter_best_tour = None
            iter_best_cost = float('inf')
            iter_best_physical = None

            # 1. Construction Phase
            # Each ant builds a permutation of cities.
            tours = []
            for _ in range(self.n_ants):
                tours.append(self._construct_tour())

            # 2. Evaluation Phase (The "Split")
            # We map the permutation (Genotype) to a set of feasible trips (Phenotype).
            for tour in tours:
                cost_est, logical_split = self._split_algorithm(tour)

                # Keep track of the best ant in this generation
                if cost_est < iter_best_cost:
                    iter_best_cost = cost_est
                    iter_best_tour = tour
                    # Only reconstruct the physical path for the winner to save CPU
                    iter_best_physical = self._reconstruct_physical_path(logical_split)

            # 3. Final Polish (Beta Optimizer)
            # If we found a good path, let's use the analytical tool to refine the trip counts.
            if iter_best_physical and path_optimizer:
                try:
                    refined_path = path_optimizer(iter_best_physical, self.problem)
                    refined_cost = self.problem.path_cost(refined_path)

                    if refined_cost < iter_best_cost:
                        iter_best_cost = refined_cost
                        iter_best_physical = refined_path
                except Exception:
                    pass  # Keep the original split if optimization fails

            # 4. Global Update
            if iter_best_cost < best_cost:
                best_cost = iter_best_cost
                best_solution_path = iter_best_physical

            # 5. Pheromone Update
            # Evaporation: Decrease all trails to make old choices irrelevant over time.
            self.pheromone *= (1.0 - self.rho)

            # Deposit: Reinforce the path chosen by the iteration's best ant.
            if iter_best_tour:
                # Normalizing Q by cost ensures shorter paths get more pheromone.
                delta = self.Q / max(iter_best_cost, 1.0)
                for k in range(len(iter_best_tour) - 1):
                    i, j = iter_best_tour[k], iter_best_tour[k + 1]
                    self.pheromone[i, j] += delta
                    self.pheromone[j, i] += delta  # Symmetric TSP assumption

        return best_solution_path, best_cost

    def _construct_tour(self):
        """
        Builds a valid permutation of all target cities.
        """
        unvisited = list(range(self.n_cities))

        # Start from a random city to ensure diversity
        start_node_idx = np.random.choice(unvisited)
        tour = [start_node_idx]
        unvisited.remove(start_node_idx)

        current = start_node_idx

        while unvisited:
            # Calculate probability distribution for the next step
            probs = self._calculate_probs(current, unvisited)

            # Roulette Wheel Selection
            next_node = np.random.choice(unvisited, p=probs)

            tour.append(next_node)
            unvisited.remove(next_node)
            current = next_node

        return tour

    def _calculate_probs(self, current, unvisited):
        """
        Standard ACO transition rule:
        Prob = (Pheromone^alpha) * (Heuristic^beta) / Sum(...)
        """
        pheromones = self.pheromone[current, unvisited]
        heuristics = self.heuristic[current, unvisited]

        numerator = (pheromones ** self.aco_alpha) * (heuristics ** self.aco_beta)
        s = numerator.sum()

        if s == 0:
            return np.ones(len(unvisited)) / len(unvisited)
        return numerator / s

    def _split_algorithm(self, tour_indices: List[int]) -> Tuple[float, List[Tuple[int, float]]]:
        """
        Prins' Split Algorithm adapted with Hard Pruning.

        Standard ACO solves TSP. Here we have a VRP (Vehicle Routing).
        This function takes the 'Giant Tour' (sequence of all cities) and finds
        the optimal points to return to the depot to unload gold.
        """
        n = len(tour_indices)
        # Bellman equation structures: V[i] is min cost to service first i cities
        V = [float('inf')] * (n + 1)
        P = [0] * (n + 1)  # Predecessors for path reconstruction
        V[0] = 0

        tour_nodes = [self.idx_to_node[idx] for idx in tour_indices]
        alpha = self.problem.alpha
        beta = self.problem.beta

        # Optimization: Don't look too far ahead if Beta is huge.
        # Long trips are impossibly expensive anyway.
        max_lookahead = n if beta < 1.5 else 5

        for i in range(n):
            if V[i] == float('inf'): continue

            current_weight = 0.0
            trip_cost = 0.0

            # Start a new trip: Depot -> First Node
            first_node = tour_nodes[i]
            trip_cost += self.shortest_dists[0][first_node]
            current_weight += self.problem.graph.nodes[first_node]['gold']

            # Try to extend this trip to subsequent nodes
            limit = min(n + 1, i + 1 + max_lookahead)

            for j in range(i + 1, limit):
                last_node_idx = j - 1
                curr_node = tour_nodes[last_node_idx]

                if j > i + 1:
                    prev_node = tour_nodes[last_node_idx - 1]
                    dist = self.shortest_dists[prev_node][curr_node]

                    # If beta >= 2, carrying weight is very expensive.
                    # If the move cost is significantly higher than just the distance,
                    # we assume it's better to cut the trip here.
                    if beta >= 2.0 and current_weight > 0:
                        estimated_move_cost = dist + (alpha * dist * current_weight) ** beta
                        if estimated_move_cost > 2.5 * dist:
                            break  # Stop extending, force a split

                    trip_cost += dist + (alpha * dist * current_weight) ** beta
                    current_weight += self.problem.graph.nodes[curr_node]['gold']

                # Calculate cost to close the trip: Node -> Depot
                d_return = self.shortest_dists[curr_node][0]
                return_cost = d_return + (alpha * d_return * current_weight) ** beta

                total_trip_cost = trip_cost + return_cost

                if V[i] + total_trip_cost < V[j]:
                    V[j] = V[i] + total_trip_cost
                    P[j] = i

        # Reconstruct the logical path (Trips)
        logical_path = []
        curr = n
        trips = []
        while curr > 0:
            prev = P[curr]
            trips.append(tour_nodes[prev:curr])
            curr = prev
        trips.reverse()

        # Flatten to the requested format: (Node, Gold)
        full_logical = []
        for trip in trips:
            full_logical.append((0, 0.0))
            for node in trip:
                gold = self.problem.graph.nodes[node]['gold']
                full_logical.append((node, gold))
        full_logical.append((0, 0.0))

        return V[n], full_logical

    def _reconstruct_physical_path(self, logical_path: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
        """
        Ensures Feasibility on sparse graphs.
        If the tour says A -> B but no edge exists, we insert the Dijkstra path.
        """
        physical_path = []
        physical_path.append(logical_path[0])

        for k in range(len(logical_path) - 1):
            u, _ = logical_path[k]
            v, v_gold = logical_path[k + 1]

            if u == v: continue

            if self.problem.graph.has_edge(u, v):
                physical_path.append((v, v_gold))
            else:
                # Fill the gap with nodes from the shortest path
                path_nodes = self.shortest_paths[u][v]
                for node in path_nodes[1:]:
                    # Intermediate nodes are just for transit (gold=0)
                    g = v_gold if node == v else 0.0
                    physical_path.append((node, g))

        return physical_path

