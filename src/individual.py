import random
import networkx as nx

class Individual:
    def __init__(self, genome, problem, dist_cache):
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
        Ricostruisce il percorso usando NetworkX. 
        Lento? Sì, ma viene chiamato 1 volta sola alla fine, quindi è irrilevante.
        """
        path = [0]
        current_node = 0
        current_weight = 0
        alpha = self.problem.alpha
        beta = self.problem.beta
        graph = self.problem.graph # Uso diretto del grafo

        for next_node in self.genome:
            gold = self.problem.graph.nodes[next_node]['gold']
            
            d_dir = self.dist_cache[current_node][next_node]
            d_home = self.dist_cache[current_node][0]
            d_out = self.dist_cache[0][next_node]

            cost_direct = d_dir + (alpha * d_dir * current_weight) ** beta
            cost_split = (d_home + (alpha * d_home * current_weight) ** beta) + \
                         (d_out + (alpha * d_out * 0) ** beta)

            if cost_split < cost_direct:
                # Torna a casa e poi vai
                path_to_home = nx.shortest_path(graph, current_node, 0, weight='dist')
                path_to_next = nx.shortest_path(graph, 0, next_node, weight='dist')
                path.extend(path_to_home[1:])
                path.extend(path_to_next[1:])
                current_weight = 0
            else:
                # Vai diretto
                path_direct = nx.shortest_path(graph, current_node, next_node, weight='dist')
                path.extend(path_direct[1:])
            
            current_weight += gold
            current_node = next_node
        
        # Ritorno finale
        path_home = nx.shortest_path(graph, current_node, 0, weight='dist')
        path.extend(path_home[1:])
        self.phenotype = path
        return path

    def mutate(self, mutation_rate=0.1):
        if random.random() < mutation_rate:
            idx1, idx2 = random.sample(range(len(self.genome)), 2)
            if idx1 > idx2:
                idx1, idx2 = idx2, idx1
            self.genome[idx1:idx2+1] = self.genome[idx1:idx2+1][::-1]