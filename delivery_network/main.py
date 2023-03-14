from graph import *

data_path = "input/"
file_name = "network.7.in"
writing_path = "output/"

"""
g = graph_from_file(data_path + file_name)
start = time.perf_counter()
g.process_kruskal()
end = time.perf_counter()
print(end-start)
"""

def get_all_min_power(files_numbers):
    for i in files_numbers:
        file_name = f"network.{i}.in"
        routes_name = f"routes.{i}.in"
        g = graph_from_file(data_path + file_name)

        file = open(data_path + routes_name, 'r')
        paths = open(writing_path + f"paths.{i}.in", 'w')
        nb_travel = int(file.readline())
        for _ in range(nb_travel):
            n1, n2, u = file.readline().split()
            node1 = int(n1) ; node2 = int(n2)
            path, power = g.min_power_mst(node1, node2)
            paths.write(f"Chemin de {node1} a {node2} : {path}, avec une puissance minimale necessaire de {power}\n")
        file.close()

files = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
start = time.perf_counter_ns()
get_all_min_power(files)
end = time.perf_counter_ns()

print(convertit(end-start, 's'))