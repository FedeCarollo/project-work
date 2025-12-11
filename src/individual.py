import random

class Individual:
    def __init__(self, genome, problem, dist_cache, path_cache):
        self.genome = genome            # Lista di città (es. [5, 2, 9...])
        self.problem = problem
        self.dist_cache = dist_cache    # Cache distanze (numeri)
        self.path_cache = path_cache    # Cache percorsi (liste nodi)
        self.fitness = float('inf')
        self.phenotype = []             # Il percorso finale completo
    def evaluate(self):
        """
        Calcola il costo usando SOLO la matematica e la cache (veloce).
        Non costruisce la lista completa dei nodi qui per risparmiare tempo.
        """
        current_node = 0
        current_weight = 0
        total_cost = 0
        
        alpha = self.problem.alpha
        beta = self.problem.beta
        
        # Scorriamo il genoma (l'ordine delle città da visitare)
        for next_node in self.genome:
            gold = self.problem.graph.nodes[next_node]['gold']
            
            # Recuperiamo le distanze dalla cache (O(1))
            # d_dir: Distanza Diretta (Current -> Next)
            # d_home: Distanza Ritorno (Current -> 0)
            # d_out: Distanza Andata (0 -> Next)
            
            try:
                d_dir = self.dist_cache[current_node][next_node]
                d_home = self.dist_cache[current_node][0]
                d_out = self.dist_cache[0][next_node]
            except KeyError:
                # Caso raro: grafo non connesso (ma il problema garantisce connessione)
                self.fitness = float('inf')
                return float('inf')

            # --- FORMULA DEL COSTO ---
            # Cost = dist + (alpha * dist * weight)^beta
            
            # 1. Ipotesi: Vado Diretto
            cost_direct = d_dir + (alpha * d_dir * current_weight) ** beta

            # 2. Ipotesi: Torno a Casa (scarico) e poi Vado
            # Costo ritorno (con peso attuale)
            c_return = d_home + (alpha * d_home * current_weight) ** beta
            # Costo andata (con peso 0)
            c_go = d_out + (alpha * d_out * 0) ** beta 
            
            cost_split = c_return + c_go

            # --- DECISIONE GREEDY ---
            if cost_split < cost_direct:
                total_cost += cost_split
                current_weight = 0 # Ho scaricato
            else:
                total_cost += cost_direct
            
            # Arrivato a destinazione, prendo l'oro
            current_weight += gold
            current_node = next_node
        
        # Ritorno finale obbligatorio a 0
        d_end = self.dist_cache[current_node][0]
        total_cost += d_end + (alpha * d_end * current_weight) ** beta

        self.fitness = total_cost
        return total_cost

    def rebuild_phenotype(self):
        """
        Ricostruisce il percorso completo (lista di nodi) usando path_cache.
        Da chiamare SOLO alla fine sul vincitore.
        """
        path = [0]
        current_node = 0
        current_weight = 0
        alpha = self.problem.alpha
        beta = self.problem.beta

        for next_node in self.genome:
            gold = self.problem.graph.nodes[next_node]['gold']
            
            d_dir = self.dist_cache[current_node][next_node]
            d_home = self.dist_cache[current_node][0]
            d_out = self.dist_cache[0][next_node]

            cost_direct = d_dir + (alpha * d_dir * current_weight) ** beta
            cost_split = (d_home + (alpha * d_home * current_weight) ** beta) + \
                         (d_out + (alpha * d_out * 0) ** beta)

            if cost_split < cost_direct:
                # Split: Aggiungi path Current->0 e 0->Next
                # [1:] per evitare di duplicare il nodo di partenza
                path.extend(self.path_cache[current_node][0][1:])
                path.extend(self.path_cache[0][next_node][1:])
                current_weight = 0
            else:
                # Direct: Aggiungi path Current->Next
                path.extend(self.path_cache[current_node][next_node][1:])
            
            current_weight += gold
            current_node = next_node
        
        # Ritorno finale
        path.extend(self.path_cache[current_node][0][1:])
        self.phenotype = path
        return path

    def mutate(self, mutation_rate=0.1):
        """Mutazione 2-OPT (Inversion)"""
        if random.random() < mutation_rate:
            idx1, idx2 = random.sample(range(len(self.genome)), 2)
            if idx1 > idx2:
                idx1, idx2 = idx2, idx1
            # Inverte la sottosequenza
            self.genome[idx1:idx2+1] = self.genome[idx1:idx2+1][::-1]