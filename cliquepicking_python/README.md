# cliquepicking

You can install the cliquepicking module with ```pip install cliquepicking```. 

The module provides the functions

- ```mec_size(G)```, which outputs the number of DAGs in the MEC represented by CPDAG G
- ```mec_sample_dags(G, k)```, which returns k uniformly sampled DAGs from the MEC represented by CPDAG G
- ```mec_sample_orders(G, k)``` which returns topological orders of k uniformly sampled DAGs from the MEC represented by CPDAG G

Be aware that ```mec_sample_dags(G, k)``` holds (and returns) k DAGs in memory. (For large graphs) to avoid large memory usage, generate DAGs in smaller batches or use ```mec_sample_orders(G, k)```, which only returns the often much smaller topological order. 

In all cases, G should be given as a edge list (vertices should be represented by zero indexed integers), which includes ```(a, b)``` and ```(b, a)``` for undirected edges $$a - b$$ and only ```(a, b)``` for directed edges $$a \rightarrow b$$. E.g.

```python
import cliquepicking as cp

edges = [(0, 1), (1, 0), (1, 2), (2, 1), (0, 3), (2, 3)]
print(cp.mec_size(edges))
```

computes the MEC size for the graph with edges $0 - 1 - 2$ and $0 \rightarrow 3 \leftarrow 2$ and hence the code should print out ```3```.

For a more involved example (Example 3 from the AAAI paper), which illustrates use with networkx, see:

```python
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
```
