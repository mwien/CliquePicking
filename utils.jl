using Graphs

"""
    readgraph(file = stdin, undirected = false)

Read a graph from the standard input or a given file and return a directed simple graph.
In case undirected graphs are read (as in the toy examples we give
here), the argument undirected=true may be passed. Then, each edge can be given only once.

In practice, for partially directed graphs like CPDAGs, an undirected
edge is expected to be encoded as two directed edges.

# Examples
```julia-repl
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
"""
function readgraph(file = stdin, undirected = false)
    if file != stdin
        file = open(file, "r")
    end
    (n, m) = parse.(Int, split(readline(file)))
    readline(file)
    G = SimpleDiGraph(n)
    for i = 1:m
        (a, b) = parse.(Int, split(readline(file)))
        add_edge!(G, a, b)
        undirected && add_edge!(G, b, a)
    end
    return G
end


"""
    ischordal(g)

Return true if the given graph is chordal
"""
function ischordal(G)
    mcsorder, invmcsorder = mcs(G)
    
    n = length(mcsorder)
    
    f = zeros(Int, n)
    index = zeros(Int, n)
    for i=n:-1:1
        w = mcsorder[i]
        f[w] = w
        index[w] = i
        for v in neighbors(G, w)
            if invmcsorder[v] > i
                index[v] = i
                if f[v] == v
                    f[v] = w
                end
            end
        end
        for v in neighbors(G, w)
            if invmcsorder[v] > i
                if index[f[v]] > i
                    return false
                end
            end
        end
    end
    return true
end
