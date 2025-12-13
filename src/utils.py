from typing import List, Tuple
import networkx as nx

def check_feasibility(
    problem,
    solution: List[Tuple[int, float]],
) -> bool:
    """
    Checks if a solution is feasible:
    1. Each step must be between adjacent cities
    2. All gold from all cities must be collected (at least once)
    
    :param problem: Problem instance
    :param solution: List of (city, gold_picked)
    :return: True if feasible, False otherwise
    """
    graph = problem.graph
    gold_at = nx.get_node_attributes(graph, "gold")
    
    # Track collected gold per city
    gold_collected = {}
    prev_city = 0  # Start from depot
    
    for city, gold in solution[1:]:
        # Check adjacency
        if not graph.has_edge(prev_city, city):
            print(f"❌ Feasibility failed: no edge between {prev_city} and {city}")
            return False
        
        # Track collected gold
        if gold > 0:
            gold_collected[city] = gold_collected.get(city, 0.0) + gold
        
        prev_city = city
    
    # Verify all gold was collected
    for city in graph.nodes():
        if city == 0:  # Depot has no gold
            continue
        expected_gold = gold_at.get(city, 0.0)
        collected_gold = gold_collected.get(city, 0.0)
        
        if abs(expected_gold - collected_gold) > 1e-6:  # Float tolerance
            print(f"❌ Feasibility failed: city {city} has {expected_gold:.2f} gold, collected {collected_gold:.2f}")
            return False
    
    return True