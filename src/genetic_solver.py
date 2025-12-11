import random
import copy
import networkx as nx
from .individual import Individual

class GeneticSolver:
    def __init__(self, problem, pop_size=100, generations=100, mutation_rate=0.3, elite_size=2):
        self.problem = problem
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        # --- OTTIMIZZAZIONE CACHE ---
        # Calcoliamo TUTTE le distanze all'avvio. 
        # Questo trasforma Dijkstra (lento) in una lettura di dizionario (istantanea).
        self.dist_cache = dict(nx.all_pairs_dijkstra_path_length(problem.graph, weight='dist'))
        self.path_cache = dict(nx.all_pairs_dijkstra_path(problem.graph, weight='dist'))

        # Città da visitare (tutte tranne 0)
        self.cities = list(problem.graph.nodes)
        if 0 in self.cities:
            self.cities.remove(0)
        
    def create_individual(self):
        genome = random.sample(self.cities, len(self.cities))
        # Passiamo le cache all'individuo
        ind = Individual(genome, self.problem, self.dist_cache, self.path_cache)
        ind.evaluate()
        return ind

    def crossover(self, p1, p2):
        """Order Crossover (OX1)"""
        size = len(p1.genome)
        start, end = sorted(random.sample(range(size), 2))
        
        child_genome = [None] * size
        child_genome[start:end+1] = p1.genome[start:end+1]
        
        current_idx = (end + 1) % size
        p2_idx = (end + 1) % size
        
        while None in child_genome:
            gene = p2.genome[p2_idx]
            if gene not in child_genome:
                child_genome[current_idx] = gene
                current_idx = (current_idx + 1) % size
            p2_idx = (p2_idx + 1) % size
            
        return Individual(child_genome, self.problem, self.dist_cache, self.path_cache)

    def evolve(self):
        # 1. Popolazione Iniziale
        population = [self.create_individual() for _ in range(self.pop_size)]
        best_overall = min(population, key=lambda x: x.fitness)
        
        # 2. Ciclo Evolutivo
        for g in range(self.generations):
            # Sort
            population.sort(key=lambda x: x.fitness)
            
            # Aggiorna Best Global
            if population[0].fitness < best_overall.fitness:
                best_overall = copy.deepcopy(population[0])
                # print(f"Gen {g}: New Best {best_overall.fitness:.2f}")

            # Elitismo
            new_population = population[:self.elite_size]
            
            # Generazione Prole
            while len(new_population) < self.pop_size:
                # Tournament Selection (piccolo k=3 per mantenere diversità)
                parent1 = min(random.sample(population, 3), key=lambda x: x.fitness)
                parent2 = min(random.sample(population, 3), key=lambda x: x.fitness)
                
                child = self.crossover(parent1, parent2)
                child.mutate(self.mutation_rate)
                child.evaluate()
                
                new_population.append(child)
            
            population = new_population

        return best_overall