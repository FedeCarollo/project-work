import random
import copy
from src.individual import Individual

class GeneticSolver:
    def __init__(self, problem, pop_size=100, generations=100, mutation_rate=0.2, elite_size=2):
        self.problem = problem
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        
        # Le città da visitare (tutte eccetto lo 0)
        self.cities = list(problem.graph.nodes)
        if 0 in self.cities:
            self.cities.remove(0)

    def create_individual(self):
        genome = random.sample(self.cities, len(self.cities))
        ind = Individual(genome, self.problem)
        ind.evaluate()
        return ind

    def crossover(self, p1, p2):
        """Order Crossover (OX1)"""
        size = len(p1.genome)
        start, end = sorted(random.sample(range(size), 2))
        
        child_genome = [None] * size
        # Copia la sottosequenza dal primo genitore
        child_genome[start:end+1] = p1.genome[start:end+1]
        
        # Riempie il resto con i geni del secondo genitore mantenendo l'ordine
        current_idx = (end + 1) % size
        p2_idx = (end + 1) % size
        
        while None in child_genome:
            gene = p2.genome[p2_idx]
            if gene not in child_genome:
                child_genome[current_idx] = gene
                current_idx = (current_idx + 1) % size
            p2_idx = (p2_idx + 1) % size
            
        ind = Individual(child_genome, self.problem)
        # La fitness verrà calcolata lazy o alla creazione della nuova popolazione
        return ind

    def evolve(self):
        # 1. Inizializzazione
        population = [self.create_individual() for _ in range(self.pop_size)]
        best_overall = min(population, key=lambda x: x.fitness)
        # 2. Loop Generazionale
        for g in range(self.generations):
            # Ordina per fitness (minore è meglio)
            population.sort(key=lambda x: x.fitness)
            
            # Elitismo: mantieni i migliori
            new_population = population[:self.elite_size]
            
            # Generazione prole
            while len(new_population) < self.pop_size:
                # Tournament Selection
                parent1 = min(random.sample(population, 5), key=lambda x: x.fitness)
                parent2 = min(random.sample(population, 5), key=lambda x: x.fitness)
                
                # Crossover
                child = self.crossover(parent1, parent2)
                
                # Mutazione
                child.mutate(self.mutation_rate)
                
                # Valutazione
                child.evaluate()
                new_population.append(child)
            
            population = new_population
            
            # Traccia il migliore globale
            current_best = population[0]
            if current_best.fitness < best_overall.fitness:
                best_overall = copy.deepcopy(current_best)
                # Logging opzionale
                # print(f"Gen {g}: New Best Cost {best_overall.fitness:.2f}")

        return best_overall