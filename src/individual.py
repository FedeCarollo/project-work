import random
import numpy as np
import networkx as nx

class Individual:
    def __init__(self, genome, problem):
        self.genome = genome  # Lista di interi (città)
        self.problem = problem
        self.fitness = float('inf')
        self.phenotype = [] # Il percorso reale con i ritorni a 0

    def evaluate(self):
        """
        Decodifica il genotipo in un percorso reale (fenotipo) usando un'euristica
        Greedy per gestire il peso e calcola il costo totale.
        """
        path = [0] # Si parte sempre da 0
        current_weight = 0
        total_cost = 0
        current_node = 0

        # Scorre le città nell'ordine stabilito dal genoma
        for next_node in self.genome:
            gold = self.problem.graph.nodes[next_node]['gold']
            
            # Calcola il costo per andare diretto
            # Nota: usiamo shortest_path per robustezza, se il grafo non è completo
            try:
                path_to_next = nx.shortest_path(self.problem.graph, current_node, next_node, weight='dist')
                cost_direct = self.problem.cost(path_to_next, current_weight)
            except nx.NetworkXNoPath:
                cost_direct = float('inf')

            # Calcola il costo per tornare a 0 (scaricare) e poi andare
            try:
                path_to_0 = nx.shortest_path(self.problem.graph, current_node, 0, weight='dist')
                path_0_to_next = nx.shortest_path(self.problem.graph, 0, next_node, weight='dist')
                
                cost_return = self.problem.cost(path_to_0, current_weight)
                cost_from_0 = self.problem.cost(path_0_to_next, 0) # Peso azzerato
                cost_split = cost_return + cost_from_0
            except nx.NetworkXNoPath:
                cost_split = float('inf')

            # Decisione Greedy: conviene tornare a casa?
            if cost_split < cost_direct:
                # Aggiungi il ritorno a 0 al percorso
                path.extend(path_to_0[1:]) # [1:] per non duplicare il nodo corrente
                path.extend(path_0_to_next[1:])
                current_weight = 0 # Scarico
                total_cost += cost_split
            else:
                # Vado diretto
                path.extend(path_to_next[1:])
                total_cost += cost_direct

            # Aggiorno stato
            current_node = next_node
            current_weight += gold
        
        # Ritorno finale obbligatorio a 0
        try:
            path_home = nx.shortest_path(self.problem.graph, current_node, 0, weight='dist')
            total_cost += self.problem.cost(path_home, current_weight)
            path.extend(path_home[1:])
        except nx.NetworkXNoPath:
            total_cost = float('inf')

        self.fitness = total_cost
        self.phenotype = path
        return total_cost

    def mutate(self, mutation_rate=0.1):
        """Applica mutatione Inversion (2-opt)"""
        if random.random() < mutation_rate:
            idx1, idx2 = random.sample(range(len(self.genome)), 2)
            if idx1 > idx2:
                idx1, idx2 = idx2, idx1
            # Inverte la sottosequenza
            self.genome[idx1:idx2+1] = self.genome[idx1:idx2+1][::-1]