data_path = "input/"

#### Définition classe unionFind pour la Q12

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n+1))
        self.rank = [0] * (n+1)
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return False
    
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y

        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x

        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1



#### Définition du graphe

class Graph:
    """
    A class representing graphs as adjacency lists and implementing various algorithms on the graphs. Graphs in the class are not oriented. 
    Attributes: 
    -----------
    nodes: NodeType
        A list of nodes. Nodes can be of any immutable type, e.g., integer, float, or string.
        We will usually use a list of integers 1, ..., n.
    graph: dict
        A dictionnary that contains the adjacency list of each node in the form
        graph[node] = [(neighbor1, p1, d1), (neighbor1, p1, d1), ...]
        where p1 is the minimal power on the edge (node, neighbor1) and d1 is the distance on the edge
    nb_nodes: int
        The number of nodes.
    nb_edges: int
        The number of edges. 
    """

    def __init__(self, nodes=[]):
        """
        Initializes the graph with a set of nodes, and no edges. 
        Parameters: 
        -----------
        nodes: list, optional
            A list of nodes. Default is empty.
        """
        self.nodes = nodes
        self.graph = dict([(n, []) for n in nodes])
        self.nb_nodes = len(nodes)
        self.nb_edges = 0

        # attributs relatifs à l'arbre couvrant (pas encore existants)
        self.mst = None
        self.mst_parents = None 
        self.mst_depths = None
    

    def __str__(self):
        """Prints the graph as a list of neighbors for each node (one per line)"""
        if not self.graph:
            output = "The graph is empty"            
        else:
            output = f"The graph has {self.nb_nodes} nodes and {self.nb_edges} edges.\n"
            for source, destination in self.graph.items():
                output += f"{source}-->{destination}\n"
        return output

#### Question 1 : Ajouter une arête au graphe
    
    def add_edge(self, node1, node2, power_min, dist=1):
        """
        Adds an edge to the graph. Graphs are not oriented, hence an edge is added to the adjacency list of both end nodes. 

        Parameters: 
        -----------
        node1: NodeType
            First end (node) of the edge
        node2: NodeType
            Second end (node) of the edge
        power_min: numeric (int or float)
            Minimum power on this edge
        dist: numeric (int or float), optional
            Distance between node1 and node2 on the edge. Default is 1.
        """
        
        if node1 not in self.graph:
            self.graph[node1] = []
            self.nb_nodes += 1
            self.nodes.append(node1)
        if node2 not in self.graph:
            self.graph[node2] = []
            self.nb_nodes += 1
            self.nodes.append(node2)

        self.graph[node1].append((node2, power_min, dist))
        self.graph[node2].append((node1, power_min, dist))
        self.nb_edges += 1

        # après ajout d'une arête, l'ancien arbre couvrant n'est plus valide
        self.mst = None
        self.mst_parents = None 
        self.mst_depths = None

# Si l'un des noeuds (node1 ou node2) n'est pas encore présent dans le graphe, il est ajouté. 
# Puis, l'arête est ajoutée aux listes d'adjacence des deux noeuds node1 et node2 avec ses caractéristiques telles que le noeud adjacent, la puissance minimale et la distance.


#### Question 2 : Liste de listes contenant les composantes connexes du graphe

    def connected_components(self):
        """Returns a list with the connected components of the graph."""
        seen = []                           # Pour stocker les noeuds visités pendant la recherche en profondeur (dfs)
        components = []                     # Pour stocker les différentes composantes connexes du graphe

        # Fonction récursive dfs

        def dfs(node, visited=None):
            if visited is None:
                visited = [node]
                seen.append(node)
            
            if node not in visited:
                visited.append(node)
                seen.append(node)

            to_visit = [n for n, _, _ in self.graph[node] if n not in visited]

            for node in to_visit:
                dfs(node, visited)
            
            return visited
        
        for i in self.nodes:
            if i not in seen:
                components.append(dfs(i))

        return components

# La fonction dfs prend en argument un noeud et une liste visited optionnelle qui stocke les noeuds visités pendant la recherche en profondeur. 
# La fonction commence par ajouter le noeud à la liste visited si elle est vide, et ajoute également le noeud à la liste seen.
# Ensuite, pour chaque noeud adjacent qui n'a pas encore été visité (n not in visited), la fonction appelle récursivement dfs sur ce noeud et met à jour la liste visited.

# La méthode connected_components() visite tous les noeuds du graphe, et si un noeud n'a pas encore été visité (i not in seen), la fonction appelle dfs sur ce noeud et stocke la liste des noeuds visités dans components.

#### Question 2 : 

    def connected_components_set(self):
        """
        The result should be a set of frozensets (one per component), 
        For instance, for network01.in: {frozenset({1, 2, 3}), frozenset({4, 5, 6, 7})}
        """
        return set(map(frozenset, self.connected_components()))

# La méthode utilise la méthode connected_components() pour obtenir les composantes connexes du graphe.
# La fonction frozenset() est appliquée à chaque sous-liste pour obtenir un ensemble non modifiable de chaque composante connexe. 
# Enfin, la fonction set() permet d'obtenir un ensemble d'ensembles de composantes connexes.

# La complexité de l'algorithme connected_components_set() dépend de la complexité de connected_components().
# Tout d'abord, connected_components() utilise une recherche en profondeur (dfs) donc dans le pire des cas, la complexité est O(V+E).
# Ensuite, connected_components_set() utilise trois fonctions : connected_components(), frozenset() et set().
# frozenset() et set() ont une complexité en O(V) au maximum.
# Ainsi, la complexité de connected_components_set() est O(V+E).


#### Question 3

    def get_path_with_power(self, src, dest, power):
        """
        Returns, if possible, a path from src to dest using only edges with a min_power less than the power of the truck.
        Else, returns None.

        Parameters:
        src : NodeType (first node of the path)
        dest : NodeType (last node of the path)
        power : power of the truck trying to get from src to dest
        """
        # set of visited nodes (avoid cycles)
        visited = set()
        
        path = self.dfs_power(src, dest, power, visited)
        
        return path
        
    def dfs_power(self, curr, dest, power, visited):
        if curr == dest:
            return [curr]
        
        visited.add(curr)

        for neighbor, min_power, _ in self.graph[curr]: # explores all neighbors of the current node

            if min_power <= power:

                if neighbor not in visited:
                    # recursively calls itself on the neighbor
                    path = self.dfs_power(neighbor, dest, power, visited)
                    # if a path was found, add the current node to the path and return it
                    if path is not None:
                        return [curr] + path
        # if we couldn't find a path to the destination, return None
        return None
    
# Cette première méthode utilise une recherche en profondeur (dfs) pour trouver un chemin entre deux noeuds (src et dest), tout en respectant la condition de puissance (power).
# Elle explore récursivement les noeuds adjacents et continue la recherche jusqu'à trouver un chemin valide vers le noeud de destination (dest).
# Si un chemin est trouvé, il est renvoyé sous la forme d'une liste de noeuds, où le premier élément est src et le dernier élément est dest.

# La complexité de l'algorithme get_path_with_power est en O(E + V)


#### Question 5 : Nouvelle méthode grâce à l'algorithme de Dijkstra

    def get_path_with_power_best(self, src, dest, power):
        """Uses Dijkstra's algorithm to return the shortest path available with a given power"""

        possible_nodes = []                     # noeuds atteignables à partir de src avec la puissance donnée
        
        def update_possible(node):            
            if node not in possible_nodes:
                possible_nodes.append(node)
            
            to_visit = [n for n, _, pow in self.graph[node] if pow <= power and n not in possible_nodes]
            for node in to_visit:
                update_possible(node)
        update_possible(src)                    # met à jour la liste des noeuds atteignables avec power

        if dest not in possible_nodes:
            return None
    
        visites = [] ; s = src

        dij = dict([(n, (float('inf'), )) for n in possible_nodes])
        # Le dictionnaire dij contient pour chaque noeud dans possible_nodes la distance la plus courte connue jusqu'à présent pour atteindre ce noeud et le noeud précédent qui donne cette distance la plus courte. 
        dij[s] = (0, None)

        while len(visites) < len(possible_nodes):
            min_dist = float('inf')
            next = None
            for k in possible_nodes:    
                if k not in visites and dij[k][0] < min_dist:
                    min_dist = dij[k][0] ; next = k             # on choisit le sommet non visité de plus courte distance

            s = next                                            # s est le sommet non visité le plus proche du sommet source
            visites.append(s)

            for (k, pow, dist) in self.graph[s]:
                if pow<=power and dij[s][0] + dist <= dij[k][0]: # si plus court en passant par s, on met à jour
                    dij[k] = (dij[s][0] + dist, s)
        
        path = [] ; node = dest
        while node != None:
            path.append(node) 
            node = dij[node][1]
                 
        path.reverse()
        return path

# À chaque itération, la fonction choisit le noeud non visité le plus proche de src (c'est-à-dire celui qui a la plus petite distance connue) et marque ce noeud comme visité.
# Elle met à jour la distance la plus courte connue pour tous les noeuds adjacents au noeud visité qui ont une puissance inférieure ou égale à la puissance du camion. 
# Si la distance la plus courte pour atteindre un noeud adjacent en passant par le noeud visité est plus courte que la distance la plus courte connue jusqu'à présent pour atteindre ce noeud, alors la distance la plus courte est mise à jour dans dij et le noeud précédent correspondant est mis à jour.
# Le chemin est reconstruit en partant de dest et en suivant les noeuds précédents dans le dictionnaire dij jusqu'au sommet source src.

# La complexité de l'algorithme de Dijkstra est en O(E + V*log V).


#### Question 6 : Recherche dichotomique pour trouver la plus petite puissance nécessaire

    def min_power(self, src, dest):
        """
        Should return path, min_power. 
        """
        if (path := self.get_path_with_power(src, dest, 0)) is not None:
            return path, 0

        a = 1 ; b = max(edge[1] for edges in self.graph.values() for edge in edges)

        path = None

        # Recherche dichotomique

        while b > a:
            pow = (a + b)//2
            path = self.get_path_with_power(src, dest, pow)
            if path is None:
                a = pow + 1
            else:
                b = pow

        if path is None: path = self.get_path_with_power(src, dest, a)
        return path, a
    
# Remarque : la fonction ne marche que pour les networks 1 et 2... Pour des networks plus grand, on stack overflow un peu vite à cause
# du nombre trop élevé d'arêtes. Par contre, la partie suivante (kruskal et min_power sur mst) fonctionnent très bien et sont objectivement
# meilleures que d'appliquer bêtement cette fonction à un énorme graphe.

# À chaque boucle, la fonction calcule la puissance médiane entre les deux bornes actuelles et utilise la fonction get_path_with_power pour trouver le chemin entre la source et la destination avec cette puissance. 
# Si le chemin est inexistant, elle met à jour la borne inférieure, sinon elle met à jour la borne supérieure. 
# Cette opération est répétée jusqu'à ce que la borne supérieure soit égale à la borne inférieure.
# Avec la puissance minimale trouvée, la fonction récupère le chemin avec cette puissance et le retourne avec la valeur de la puissance minimale.

# La complexité de cet algorithme est O(E * log(P)) car la recherche dichotomique a une complexité de O(log(P)) avec P la puissance requise
# et la recherche d'un chemin est en O(E) car on fait un parcours en profondeur.

#### Question 12 :

# Trouver s'il existe déjà un chemin entre le noeud 1 et le noeud 2 qui forme un cycle

    def is_cycle(self, n1, n2):
        
        def dfs(node, visited=None):
            if visited is None:
                visited = [node]
            if node not in visited:
                visited.append(node)
            to_visit = [n for n, _, _ in self.graph[node] if n not in visited]
            for node in to_visit:
                dfs(node, visited)
            return visited
        
        return n2 in dfs(n1)

# La fonction utilise une recherche en profondeur (dfs) pour explorer les noeuds du graphe à partir d'un noeud donné n1. 
# Elle commence par initialiser une liste visited qui contient le noeud n1, puis elle ajoute chaque noeud visité à cette liste. 
# Elle utilise la liste visited pour éviter de visiter à nouveau les noeuds déjà visités.


# Algorithme de Kruskal pour trouver l'arbre couvrant de poids minimum dans un graphe pondéré non orienté.
   
    def kruskal_no_uf(self):
        edges = []
        for n1 in self.nodes:
            for (n2, pow, _) in self.graph[n1]:
                if n1 < n2: edges.append((n1, n2, pow))
        
        edges = sorted(edges, key = lambda a : a[2])

        g_mst = Graph(self.nodes)
        for (n1, n2, pow) in edges:
            if self.is_cycle(n1, n2): continue
            g_mst.add_edge(n1, n2, pow)
        self.mst = g_mst
        return g_mst
    
# La fonction commence par parcourir tous les noeuds du graphe et pour chaque noeud n1, elle parcourt tous les noeuds adjacents n2 de n1 en récupérant leur poids. 
# Elle ajoute ensuite une arête de n1 à n2 avec le poids correspondant dans une liste edges, à condition que n1 < n2 pour éviter de prendre en compte deux fois la même arête.
# La fonction trie ensuite la liste edges par ordre croissant de poids en utilisant la fonction sorted().
# Pour chaque arête (n1, n2, pow), elle vérifie si l'ajout de cette arête crée un cycle en appelant la fonction is_cycle sur les noeuds n1 et n2. 
# Si c'est le cas, elle continue à la prochaine arête. Sinon, elle ajoute l'arête à g_mst en appelant la méthode add_edge.
# Ainsi, la fonction retourne le graphe g_mst qui contient l'arbre couvrant de poids minimum.

# Remarque : Sans structure UnionFind, on remarque que c'est très très lent. Voici donc une fonction kruskal adaptée :

# Avant de faire la fonction kruskal : implémentons la structure UnionFind:


    def kruskal(self):
        """
        Returns the minimum spanning tree of the graph and adds it to the graph attributes.
        """
        if self.mst is not None: return self.mst
        uf = UnionFind(self.nb_nodes) # creates the UnionFind structire, keeping track of connected components
        
        edges = [(power, src, dest) for src in self.nodes for dest, power, _ in self.graph[src] if src < dest]
        edges.sort()

        mst = Graph(self.nodes)
        for power, src, dest in edges:
            #finds the sets that contain src and dest
            src_set = uf.find(src)
            dest_set = uf.find(dest)
            if src_set != dest_set:
                mst.add_edge(src, dest, power)
                uf.union(src_set, dest_set)
        
        return mst


#### Question 14 : 

# On va créer 2 dictionnaires permettant de stocker respectivement le parent d'un noeud ainsi que sa profonderu dans l'arbre couvrant.

    def process_kruskal(self):
        if self.mst is None: self.mst = self.kruskal()
        parents = {} ; depths = {}
        
        def dfs(g, node, depth, visited = None):
            if visited is None: # pour le premier appel uniquement
                visited = set([node])
                parents[node] = None
            depths[node] = depth # on ajoute la profondeur du noeud
            for neighbor, power, dist in g.graph[node]:
                if neighbor not in  visited:
                    visited.add(neighbor) # on dit qu'on a visité le voisin pour ne pas y repasser
                    parents[neighbor] = (node, power) # on ajoute le parent du noeuf ainsi que la puissance du chemin entre les 2
                    dfs(g, neighbor, depth+1, visited) # la profondeur augmente de 1 pour chaque voisin

        dfs(self.mst, self.mst.nodes[0], 0, None) # on appelle la fonction avec pour référence le premier noeud de l'arbre
        
        self.mst_parents = parents ; self.mst_depths = depths # on met à jour les attributs du graphe g        

# ensuite, on recréera les chemins de scr et dest jusqu'à l'ancêtre commun afin de trouver l'unique chemin entre ces 2 noeuds

    def min_power_mst(self, src, dest):
        if self.mst is None: self.process_kruskal()
        g = self.mst
        path_from_src = [src] # pour reconstruire le chemin depuis src jusqu'à l'ancêtre commun
        path_from_dest = [dest] # pour reconstruire le chemin depuis dest jusqu'à l'ancêtre commun
        pow = 0
    
        while self.mst_depths[src] > self.mst_depths[dest]: # si le sommet src est plus profond que le sommet dest
            src, powsrc = self.mst_parents[src]
            path_from_src.append(src)
            pow = max(pow, powsrc)
        while self.mst_depths[src] < self.mst_depths[dest]: # pareil mais de l'autre côté
            dest, powdest = self.mst_parents[dest]
            path_from_dest.append(dest)
            pow = max(pow, powdest)
        while src != dest:
            src, powsrc = self.mst_parents[src]
            dest, powdest = self.mst_parents[dest]
            path_from_src.append(src)
            path_from_dest.append(dest)
            pow = max(pow, powsrc, powdest) # on met à jour la puissance (si besoin)

        #print(path_from_src, src, path_from_dest)
        return path_from_src[:-1] + path_from_dest[::-1], pow
        # path_from_src représente le chemin de src jusqu'à l'ancêtre commun. On y enlève le dernier terme (c'est l'ancêtre commun, qui
        # est déjà contenu dans le tableau path_from_dest, qu'on retourne pour avoir le chemin de l'ancêtre jusqu'à dest).

    
# La fonction min_power_mst() utilise l'algorithme de Kruskal. On remonte depuis les noeuds src et dest jusqu'à leur ancêtre commun dans l'arbre,
# et on reconstruit le chemin petit à petit. On concatène ensuite les chemins de src jusqu'à l'ancêtre avec celui de l'ancêtre jusqu'à dest.
# On se trouve dans un arbre donc un tel chemin est unique. L'arbre étant couvrant de poids minimal, cela garantit que le poids maximal rencontré
# est la puissance minimale nécessaire pour aller de src à dest.


# Question 15

# La complexité de l'algorithme de Kruskal est O(E * log V), où E est le nombre d'arêtes du graphe d'origine. 
# On fait ensuite un parcourt en profondeur en O(E+V) pour récupérer les parents et les profondeurs.
# Par la suite, on a juste à remonter jusqu'à l'ancêtre commun, et la profondeur est au maximum de E.
# On a donc un algorithme en O(E * logV) (car E = V-1)



#### Question 1 : Transformation d'un fichier txt en un objet de la classe Graph représentant le graphe contenu dans le fichier

def graph_from_file(filename):
    """
    Reads a text file and returns the graph as an object of the Graph class.

    The file should have the following format: 
        The first line of the file is 'n m'
        The next m lines have 'node1 node2 power_min dist' or 'node1 node2 power_min' (if dist is missing, it will be set to 1 by default)
        The nodes (node1, node2) should be named 1..n
        All values are integers.

    Parameters: 
    -----------
    filename: str
        The name of the file

    Outputs: 
    -----------
    G: Graph
        An object of the class Graph with the graph from file_name.
    """
    with open(filename, "r") as file:            # ouverture du fichier
        n, m = map(int, file.readline().split(" ")) # n noeuds et m arêtes
        g = Graph(list(range(1, n+1)))                 # graphe initialisé avec n noeuds
        for _ in range(m):
            edge = file.readline().split()
            if len(edge) == 3:
                n1, n2, pow_min = edge
                node1 = int(n1) ; node2 = int(n2) ; power = float(pow_min)
                g.add_edge(node1, node2, power) # ajoute dist=1 par défaut
            elif len(edge) == 4:
                n1, n2, pow_min, dist = edge
                node1 = int(n1) ; node2 = int(n2) ; power = float(pow_min) ; distance = float(dist)
                g.add_edge(node1, node2, power, distance)
            else:
                raise Exception("Format incorrect")
    file.close()                                  # fermeture du fichier
    return g                                      # renvoi du graphe

# La première ligne du fichier est "n m", où n est le nombre de noeuds dans le graphe et m est le nombre d'arêtes dans le graphe.
# Pour chaque ligne, la fonction lit les données de l'arête sous forme de liste d'entiers, et ajoute l'arête au graphe avec la méthode add_edge() de la classe Graph. 
# Si la longueur de la liste est 3, alors dist est absent et la méthode add_edge() utilise la valeur par défaut de dist=1. Si la longueur est 4, dist est également fourni.


#### Question 10 :

import time, random

def convertit(timer, format):
    if format == 'ns' : return timer
    if format == 's' : return timer * 10**(-9)
    if format == 'm' : return timer/60 * 10**(-9)
    if format == 'h' : return timer/3600 * 10**(-9)
    if format == 'd' : return timer/86400 * 10**(-9)

def question_10(file_number, format = 's'):
    """Returns expected time to calculate all minimum powers, in seconds"""
    g = graph_from_file(data_path + f"network.{file_number}.in")

    with open(data_path + f"routes.{file_number}.in") as file:
        nb_lines = int(file.readline())
        k = random.randint(1, nb_lines) # choix aléatoire de la ligne sur laquelle on calcule le temps
        for _ in range(k-1):
            file.readline() # on lit les k-1 première lignes

        n1, n2, _ = file.readline().split() # noeuds de la k-ième ligne
        node1 = int(n1) ; node2 = int(n2)

        start = time.perf_counter_ns()
        g.min_power(node1, node2)
        end = time.perf_counter_ns()
    file.close()
    timer = (end-start)*nb_lines
    return convertit(timer, format)


# La fonction mesure le temps total nécessaire pour traiter les k routes et calcule le temps moyen nécessaire pour trouver le chemin avec la puissance minimale pour chaque route. 
# Elle renvoie le temps moyen sous forme d'entier arrondi.

def question_14(file_number, format = 's'):
    """Returns expected time to calculate all minimum powers, in seconds"""
    g = graph_from_file(data_path + f"network.{file_number}.in")
    g.process_kruskal()
    with open(data_path + f"routes.{file_number}.in") as file:
        nb_lines = int(file.readline())
        k = random.randint(1, nb_lines) # choix aléatoire de la ligne sur laquelle on calcule le temps
        for _ in range(k-1):
            file.readline() # on lit les k-1 première lignes

        n1, n2, _ = file.readline().split() # noeuds de la k-ième ligne
        node1 = int(n1) ; node2 = int(n2)

        start = time.perf_counter_ns()
        g.min_power_mst(node1, node2)
        end = time.perf_counter_ns()
    file.close()
    timer = (end-start)*nb_lines
    return convertit(timer, format)
