# cliquepicking

You can install the cliquepicking module with ```pip install cliquepicking```. 

The module provides the functions

- ```mec_size(G)```, which outputs the number of DAGs in the MEC represented by CPDAG G
- ```mec_list_dags(G)```, which returns a list of all DAGs in the MEC represented by CPDAG G
- ```mec_list_orders(G)```, which returns topological orders of all DAGs in the MEC represented by CPDAG G

and the class ```MecSampler```, which can be constructed with ```MecSampler(G)``` and has the methods ```sample_dag()``` and ```sample_order()```, which produce a single DAG (or its topological order) from the MEC represented by CPDAG G. Repeatedly calling these methods is computationally cheap compared to initially constructing the sampler. 

The DAGs are returned as edge lists and they can be read e.g. in networkx using ```nx.DiGraph(dag)``` (see the example at the bottom).

Be aware that ```mec_sample_dags(G, k)``` holds (and returns) k DAGs in memory. (For large graphs) to avoid high memory demand, generate DAGs in smaller batches or use ```mec_sample_orders(G, k)```, which only returns the easier-to-store topological order. 

Be aware that ```mec_list_dags(G)``` holds in memory (and returns) all DAGs in the MEC. For large MECs this can lead to out-of-memory errors, so consider checking the size of the MEC using ```mec_size(G)``` before calling this method.

In all cases, G should be given as an edge list (vertices should be represented by zero-indexed integers), which includes ```(a, b)``` and ```(b, a)``` for undirected edges $a - b$ and only ```(a, b)``` for directed edges $a \rightarrow b$. E.g.

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

sampler = cp.MecSampler(list(G.edges))
for _ in range(5):
    print(nx.DiGraph(sampler.sample_dag()))

for _ in range(5):
    print(sampler.sample_order())
```

## Time Complexity
The counting procedure has worst-case run-time $n^3$, with $n$ denoting the number of vertices of the input graph. This improves the asymptotic complexity from the literature based on the comment at the bottom of page 80 in my [thesis](https://epub.uni-luebeck.de/items/44cd6c2b-c86a-40bc-aef2-669310429ac6). That is, each recursive call is associated with a subtree of the clique-tree of $G$ shaving off a factor of $n$. By using global memoization the computations of function $\phi$ are also possible in this run-time. The general procedure is identical to the published algorithm. There are a few additional practical optimizations that do not further improve the worst-case run-time but make the algorithm faster in practice compared to previous implementations. 
