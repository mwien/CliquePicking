import cliquepicking as cp
import networkx as nx

G = nx.DiGraph()
G.add_edge(0, 1)
G.add_edge(1, 0)
G.add_edge(0, 2)
G.add_edge(2, 0)
G.add_edge(1, 2)
G.add_edge(2, 1)
G.add_edge(1, 3)
G.add_edge(3, 1)
G.add_edge(1, 4)
G.add_edge(4, 1)
G.add_edge(1, 5)
G.add_edge(5, 1)
G.add_edge(2, 3)
G.add_edge(3, 2)
G.add_edge(2, 4)
G.add_edge(4, 2)
G.add_edge(2, 5)
G.add_edge(5, 2)
G.add_edge(3, 4)
G.add_edge(4, 3)
G.add_edge(4, 5)
G.add_edge(5, 4)
print(cp.mec_size(list(G.edges)))
sample_dags = cp.mec_sample_dags(list(G.edges), 5)
for dag in sample_dags:
    print(nx.DiGraph(dag))
sample_orders = cp.mec_sample_orders(list(G.edges), 5)
for order in sample_orders:
    print(order)