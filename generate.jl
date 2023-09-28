using Graphs
using Random

include("utils.jl")

function gencc(n, m, seed=123)
    Random.seed!(seed)
    G = SimpleDiGraph(n)

    # 1. Prüfer sequence of n-2 numbers, each in [1, n-2]
    ps = rand(1:n-2, n-2)

    # 2. Convert the Prüfer sequence into a tree
    deg = fill(1, n)
    for i in ps
        deg[i] += 1
    end

    for i in ps
        for j = 1:n
            if deg[j] == 1
                add_edge!(G, i, j)
                add_edge!(G, j, i)
                deg[i] -= 1
                deg[j] -= 1
                break
            end
        end
    end

    u, v = 0, 0
    for i in 1:n
        if deg[i] == 1
            if u == 0
                u = i
            else
                v = i
                break
            end
        end
    end
    add_edge!(G, u, v)
    add_edge!(G, v, u)
    deg[u] -= 1
    deg[v] -= 1

    # 3. Add remaining edges
    ecount = convert(Int, ne(G) / 2)
    while ecount < m
        u = rand(1:n)
        v = rand(1:n)
        if u != v && !has_edge(G, u, v) && !has_edge(G, v, u)
            add_edge!(G, u, v)
            add_edge!(G, v, u)
            if ischordal(G)
                ecount += 1
            else
                rem_edge!(G, u, v)
                rem_edge!(G, v, u)
            end
        end
    end

    return G
end
