import sys 
sys.path.append("delivery_network")

from graph import Graph, graph_from_file

import unittest   # The test framework

class Test_Kruskal(unittest.TestCase):
    
    def test_network(self): # test personnel
        g = graph_from_file("input/network.9.in")
        g_mst = g.kruskal()
        self.assertEqual(g_mst.nb_nodes, g.nb_nodes) # checks if the number of nodes is the same
        self.assertEqual(g_mst.nb_edges, g.nb_nodes - 1) # checks if the number of edges is correct

        # checks if the mst is a tree
        visited = set()
        q = [1]
        while len(q) > 0:
            node = q.pop(0)
            visited.add(node)
            for neighbor, _, _ in g_mst.graph[node]:
                if neighbor not in visited:
                    q.append(neighbor)
        self.assertEqual(len(visited), g.nb_nodes)

    def test_network00(self):
        g = graph_from_file("input/network.00.in")
        g_mst = g.kruskal()
        mst_expected = {1: [(8, 0, 1), (2, 11, 1), (6, 12, 1)],
                        2: [(5, 4, 1), (3, 10, 1), (1, 11, 1)],
                        3: [(4, 4, 1), (2, 10, 1)],
                        4: [(3, 4, 1), (10, 4, 1)],
                        5: [(2, 4, 1), (7, 14, 1)],
                        6: [(1, 12, 1)],
                        7: [(5, 14, 1)],
                        8: [(1, 0, 1), (9, 14, 1)],
                        9: [(8, 14, 1)],
                        10: [(4, 4, 1)]}
        self.assertEqual(g_mst.graph, mst_expected)

    def test_network05(self):
        g = graph_from_file("input/network.05.in")
        g_mst = g.kruskal()
        mst_expected = {1: [(3, 2, 1), (4, 4, 1), (2, 6, 1)],
                        2: [(1, 6, 1)],
                        3: [(1, 2, 1)],
                        4: [(1, 4, 1)],
                        }
        self.assertEqual(g_mst.graph, mst_expected)

if __name__ == '__main__':
    unittest.main()

# REMARQUE : les fonctions étant les mêmes pour chaque network, on teste directement en changeant le filename au début,
# sans changer la fonction.
