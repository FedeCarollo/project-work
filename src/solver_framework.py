# from s339239 import Problem
from src.utils import optimize_full_path
from src.beta_optimizer import path_optimizer
from src.merge_optimizer import merge_strategy_optimized
from src.utils import check_feasibility
import logging
from src.genetic_solver import GeneticSolver
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import time


def problem_solver(problem) -> tuple[list[tuple[int, float]], float]:
    """
    Multithreaded solver that runs multiple strategies in parallel and selects the best solution.
    """
    
    # Dictionary of available solvers - easily extensible for future solvers
    solvers = {
        'Genetic': genetic_solver,
        'Merge': merge_solver,
    }
    
    # Run all solvers in parallel
    results = {}
    with ThreadPoolExecutor(max_workers=len(solvers)) as executor:
        futures = {executor.submit(solver_func, problem): name 
                   for name, solver_func in solvers.items()}
        
        for future in as_completed(futures):
            solver_name = futures[future]
            path, cost = future.result()
            results[solver_name] = (path, cost)
    
    # Log all solver costs
    costs_summary = " | ".join([f"{name}: {cost:.2f} | feasible: {check_feasibility(problem, path)}" for name, (path, cost) in results.items()])
    logging.debug(f"All solver costs: {costs_summary}")
    
    # Select best solution
    best_solver = min(results.items(), key=lambda x: x[1][1])
    best_name, (best_path, best_cost) = best_solver
    
    logging.info(f"Selected {best_name} solution with cost: {best_cost:.2f}")
    return best_path, best_cost




def genetic_solver(problem) -> float:
    start_time = time()
    
    # GA parameters (can be increased for better results, e.g., pop=200, gen=500)
    POPULATION_SIZE = 10
    GENERATIONS = 20
    MUTATION_RATE = 0.3
    ELITE_SIZE = 3

    # Initialize and run the solver
    solver = GeneticSolver(
        problem=problem, 
        pop_size=POPULATION_SIZE,
        generations=GENERATIONS,
        mutation_rate=MUTATION_RATE,
        elite_size=ELITE_SIZE
    )
    
    best_individual = solver.evolve()
    
    # Extract final solution (path and cost)
    path = best_individual.rebuild_phenotype()
    cost = best_individual.fitness
    

    path = [(0, 0.0)] + path  # Ensure depot at start
    # Apply beta-optimization to full genetic path
    optimized_path = optimize_full_path(path, problem)
    optimized_cost = problem.path_cost(optimized_path)
    
    elapsed_time = time() - start_time
    logging.info(f"Genetic Solver: pop={POPULATION_SIZE}, gen={GENERATIONS} | Initial cost: {cost:.2f} → Optimized cost: {optimized_cost:.2f} | Path length: {len(optimized_path)} steps | Time: {elapsed_time:.2f}s")
    logging.debug(f"Optimized path: {optimized_path}")

    return optimized_path, optimized_cost

def merge_solver(problem) -> list[tuple[int, float]]:
    """
    Main solver function using the optimized merge strategy.
    
    Returns:
        Final path as a list of (city, gold_picked) tuples.
    """
    start_time = time()
    merged_paths = merge_strategy_optimized(problem)
    optimized_paths = [path_optimizer(trip, problem) for trip in merged_paths] if problem._beta > 1 else merged_paths
    
    # Flatten all trips into a single path
    final_path = []
    for trip in optimized_paths:
        final_path.extend(trip[:-1])  # Exclude last depot to avoid duplication
    final_path.append((0, 0.0))  # End at depot
    
    cost = problem.path_cost(final_path)
    elapsed_time = time() - start_time

    logging.info(f"Merge Solver: β={problem._beta:.2f} | Trips: {len(optimized_paths)} | Cost: {cost:.2f} | Path length: {len(final_path)} steps | Time: {elapsed_time:.2f}s")
    logging.debug(f"Final path: {final_path}")

    return final_path, cost