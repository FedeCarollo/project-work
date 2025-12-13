# from s339239 import Problem

def path_splitter(path: list[tuple[int, float]]) -> list[list[tuple[int, float]]]:
    """
    Split the path in subpaths where gold is picked up.
    """
    subpaths = []
    current_subpath = []

    starting_node = 0
    cumulated_gold = 0.0

    for (node, gold) in path:
        current_subpath.append((node, gold))
        if gold > 0:
            subpaths.append({
                'start_node': starting_node,
                'path': current_subpath,
                'cumulated_gold': cumulated_gold
            })
            current_subpath = []
            starting_node = node
            cumulated_gold += gold
    
    if current_subpath:
        subpaths.append({
            'start_node': starting_node,
            'path': current_subpath,
            'cumulated_gold': cumulated_gold
        })

    return subpaths

def compute_beta_weighted_static_cost(beta: float, path_split: dict, problem) -> float:
    """
    Compute the static cost of the path considering beta-weighted static cost.
    Just the linear part of the cost function.
    """

    src = path_split['start_node']
    path = path_split['path']
    
    cost = 0.0

    for i in range(len(path) - 1):
        dest = path[i][0]
        dist = problem.graph[src][dest]['dist']
        cost += dist**beta  # Linear part only
        src = dest

    return cost




def path_optimizer(path: list[tuple[int, float]], problem) -> list[tuple[int, float]]:
    """
    Optimize the given path by determining the optimal number of trips (N_opt)
    using the linear scan method (no path splitting).
    """
    alpha = problem._alpha
    beta = problem._beta

    # If beta <= 1, no optimization possible
    if beta <= 1:
        return path 

    total_static_dist = 0.0      # Denominator: sum_static
    total_weighted_cost = 0.0    # Numerator: sum_weighted
    
    current_gold_on_vehicle = 0.0

    # Iterate through each edge in the path
    for i in range(len(path) - 1):
        u, gold_at_u = path[i]
        v, gold_at_v = path[i+1]
        
        # 1. Get geometric distance of the edge
        dist = problem.graph[u][v]['dist']
        
        # 2. Update gold on vehicle BEFORE moving?
        # Depends on the logic: path[i] tells how much gold is at node i.
        # If we are at u, we pick up gold at u and THEN traverse the edge u->v.
        # (If gold at u is 0, nothing changes).
        current_gold_on_vehicle += gold_at_u
        
        # 3. Accumulate costs
        # Static Cost (Denominator): Only counts pure distance
        total_static_dist += dist
        
        # Weighted Cost (Numerator): (distance * weight_carried)^beta
        if current_gold_on_vehicle > 0:
            total_weighted_cost += (dist * current_gold_on_vehicle) ** beta

    # Avoid division by zero if no weighted cost
    if total_weighted_cost == 0:
        return path
    
    # 4. Apply magic formula
    term = (beta - 1) * total_weighted_cost * (alpha**beta) / total_static_dist
    N_opt = term ** (1 / beta)
    
    N_opt = int(round(N_opt))
    
    # Safety check: at least 1 trip
    if N_opt < 1: 
        N_opt = 1

    # print(f"Calculated N_opt: {N_opt}\r")

    if N_opt > 1:
        # Create fractional path
        # Note: we need to divide the gold at each node by N_opt
        fractional_path = []
        run = [(node, gold / N_opt) for (node, gold) in path]
        run.pop() # Remove last node to avoid duplication
        fractional_path = run * N_opt
        fractional_path.append((0, 0.0))  # Add depot at the end
        return fractional_path
    else:
        return path

if __name__ == "__main__":
    import logging
    import networkx as nx

    logging.basicConfig(level=logging.INFO)

    prob = Problem(num_cities=75, alpha=1.0, beta=2.0, density=0.3, seed=142, gold=1000.0)

    # Take a target node, dijkstra to get go path
    target_node = 20
    
    go_path = nx.dijkstra_path(prob.graph, source=0, target=target_node, weight='dist')

    if len(go_path) < 3:
        raise ValueError("Go path too short to optimize")

    return_path = nx.dijkstra_path(prob.graph, source=target_node, target=0, weight='dist')

    # Add gold pickup at target node and next node (if not 0)
    full_path = [(node, 0.0) for node in go_path]
    full_path.pop()  # Remove last node to avoid duplication
    # full_path.pop(0)  # Remove first node to avoid duplication
    full_path.append((target_node, prob.graph.nodes[target_node]['gold']))
    
    # Pick gold at next node
    full_path.append((return_path[1], prob.graph.nodes[return_path[1]]['gold']))
    # Add return path
    full_path.extend([(node, 0.0) for node in return_path[2:]])
    # full_path.pop()  # Remove last node to avoid duplication

    
    print("Original path:")
    print(full_path)

    
    new_path = path_optimizer(full_path, prob)

    old_cost = prob.path_cost(full_path)
    new_cost = prob.path_cost(new_path)

    print(f"Original cost: {old_cost}")
    print(f"Optimized cost: {new_cost}")
    