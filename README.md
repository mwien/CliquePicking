# The Clique-Picking Algorithm
This repository contains an implementation of the Clique-Picking algorithm that we have proposed at AAAI 2021 [1] for counting the number Markov Equivalent DAGs in polynomial time. Moreover, we give an implementation of the polynomial-time algorithm for uniformly sampling a DAG from a Markov Equivalence Class. The paper and its appendix is available on arXiv [2].

1. Marcel Wienöbst, Max Bannach, and Maciej Liśkiewicz: *Polynomial-Time Algorithms for Counting and Sampling Markov Equivalent DAGs* (AAAI 2021)
2. [arXiv version](https://arxiv.org/abs/2012.09679)

## Graph Usage

These algorithms are implementated in Julia and use the LightGraphs package. In particular, the graphs are represented as SimpleDiGraphs.
    A graph can be loaded from a file or from standard input using the readgraph function from utils.jl. The input has to have the following format: The first line contains the number of vertices n and edges m, this is followed by blank line. It ends with m lines containing the edges of the graph.

If the graph is undirected as in the example, one may give every edge
once and pass the parameter "undirected=true". Else it is expected
that undirected edges like 1-2 are given as two directed edges "1 2" and "2 1".

```julia
julia> readgraph(stdin, true)
6 11

1 2
1 3
2 3
2 4
2 5
2 6
3 4
3 5
3 6
4 5
5 6
{6, 22} directed simple Int64 graph
```

The input graph above is discussed in Example 3 and 5 in [1,2] and given in the file "example.in". It looks as follows:
<p align="center">
  <a><img width="30%" src="https://github.com/mwien/CliquePicking/raw/master/example.png" title="Example"></a>
</p>


## Compute number of Markov equivalence classes

1. Load a graph G (the parameter true is used because the graph is
    undirected, remove this for partially directed graphs).
    ```julia
    julia> G = readgraph("example.in", true)
    {6, 22} directed simple Int64 graph
    ```
2. Pass it to the MECsize function from counting.jl.
    ```julia
    julia> MECsize(G)
    54
    ```
    The number of Markov equivalence classes is 54.

## Sample a DAG

1. Load a graph G (the parameter true is used because the graph is
    undirected, remove this for partially directed graphs).
    ```julia
    julia> G = readgraph("example.in", true)
    {6, 22} directed simple Int64 graph
    ```
2. Pass it to the sampleDAG function from sampling.jl.

    ```julia
    julia> sampleDAG(G)
    {6, 11} directed simple Int64 graph
    ```
    The output is a directed graph.

    This function internally does a precomputation step before sampling. Therefore, when sampling several DAGs from one input graph G, it can be more efficient to do the precomputation only once.  In this case do
    ```julia
    julia> pre = precomputation(G);
    julia> sampleDAG(G, pre)
    {6, 11} directed simple Int64 graph
    ```

## AAAI 2021 Experiments
In [1], we used the C++ implementation for counting, which can be found under [aaai_experiments](/aaai_experiments). This folder also contains some of the graphs we used in the experiments.
   The Julia implementation is a factor <2 slower compared to the C++ code.