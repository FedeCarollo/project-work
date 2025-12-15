import random
import networkx as nx
from Problem import Problem

class Individual:
    def __init__(self, genome, problem: Problem, dist_cache):
        self.genome = genome            
        self.problem = problem
        self.dist_cache = dist_cache    
        # path_cache rimosso: non serve durante l'evoluzione!
        self.fitness = float('inf')
        self.phenotype = []             

    def evaluate(self):
        current_node = 0
        current_weight = 0
        total_cost = 0
        
        alpha = self.problem.alpha
        beta = self.problem.beta
        
        for next_node in self.genome:
            gold = self.problem.graph.nodes[next_node]['gold']
            
            try:
                d_dir = self.dist_cache[current_node][next_node]
                d_home = self.dist_cache[current_node][0]
                d_out = self.dist_cache[0][next_node]
            except KeyError:
                self.fitness = float('inf')
                return float('inf')

            # Calcolo costi (Matematica pura, niente liste)
            cost_direct = d_dir + (alpha * d_dir * current_weight) ** beta
            
            c_return = d_home + (alpha * d_home * current_weight) ** beta
            c_go = d_out + (alpha * d_out * 0) ** beta 
            cost_split = c_return + c_go

            if cost_split < cost_direct:
                total_cost += cost_split
                current_weight = 0 
            else:
                total_cost += cost_direct
            
            current_weight += gold
            current_node = next_node
        
        d_end = self.dist_cache[current_node][0]
        total_cost += d_end + (alpha * d_end * current_weight) ** beta

        self.fitness = total_cost
        return total_cost

    def rebuild_phenotype(self):
        """
        Ricostruisce il percorso come lista di tuple [(nodo, oro_preso), ...]
        """
        path = [] # Ora è una lista di tuple
        current_node = 0
        current_weight = 0
        alpha = self.problem.alpha
        beta = self.problem.beta
        graph = self.problem.graph

        for next_node in self.genome:
            gold = self.problem.graph.nodes[next_node]['gold']
            
            d_dir = self.dist_cache[current_node][next_node]
            d_home = self.dist_cache[current_node][0]
            d_out = self.dist_cache[0][next_node]

            cost_direct = d_dir + (alpha * d_dir * current_weight) ** beta
            cost_split = (d_home + (alpha * d_home * current_weight) ** beta) + \
                         (d_out + (alpha * d_out * 0) ** beta)

            if cost_split < cost_direct:
                # 1. Torna a casa (scarica)
                p_home = nx.shortest_path(graph, current_node, 0, weight='dist')
                # Aggiungi tutti i passi intermedi con 0 oro
                for step in p_home[1:-1]:
                    path.append((step, 0))
                path.append((0, 0)) # Arrivo a casa
                
                # 2. Ripartire verso next_node
                p_next = nx.shortest_path(graph, 0, next_node, weight='dist')
                for step in p_next[1:-1]:
                    path.append((step, 0))
                
                current_weight = 0
            else:
                # Vai diretto (passi intermedi)
                p_dir = nx.shortest_path(graph, current_node, next_node, weight='dist')
                for step in p_dir[1:-1]:
                    path.append((step, 0))
            
            # Arrivo alla destinazione e PRENDO L'ORO
            path.append((next_node, gold))
            
            current_weight += gold
            current_node = next_node
        
        # Ritorno finale a 0
        p_end = nx.shortest_path(graph, current_node, 0, weight='dist')
        for step in p_end[1:]:
            path.append((step, 0))
            
        self.phenotype = path
        return path

    def mutate(self, mutation_rate=0.1):
        """
        Applica mutazione con probabilità mutation_rate.
        Supporta Inversion (2-opt) e Swap.
        """
        if random.random() < mutation_rate:
            # 50% probabilità di Inversion, 50% di Swap
            if random.random() < 0.5:
                # Inversion Mutation (ottima per TSP geometrici)
                idx1, idx2 = random.sample(range(len(self.genome)), 2)
                if idx1 > idx2:
                    idx1, idx2 = idx2, idx1
                self.genome[idx1:idx2+1] = self.genome[idx1:idx2+1][::-1]
            else:
                # Swap Mutation (ottima per riordinare singole città)
                idx1, idx2 = random.sample(range(len(self.genome)), 2)
                self.genome[idx1], self.genome[idx2] = self.genome[idx2], self.genome[idx1]

    def local_optimize(self):
        """
        Algoritmo Memetico: Ricerca Locale (Hill Climbing).
        Prova a scambiare coppie di città adiacenti. Se migliora, tiene il cambiamento.
        """
        improved = False
        current_fitness = self.fitness
        
        # Per efficienza, non proviamo tutti gli scambi possibili se il genoma è enorme,
        # ma facciamo una passata lineare sugli adiacenti.
        for i in range(len(self.genome) - 1):
            # Swap adiacente
            self.genome[i], self.genome[i+1] = self.genome[i+1], self.genome[i]
            
            new_fitness = self.evaluate()
            
            if new_fitness < current_fitness:
                current_fitness = new_fitness
                improved = True
                # Manteniamo lo scambio (First Improvement)
            else:
                # Revert
                self.genome[i], self.genome[i+1] = self.genome[i+1], self.genome[i]
                self.fitness = current_fitness # Ripristina fitness precedente
                
        return improved