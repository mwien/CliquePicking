using Graphs
using LinkedLists

include("utils.jl")

struct Memo
    amos::Vector{BigInt}
    rho::Dict{Vector, BigInt}
    fac::Vector{BigInt}
end

# TODO: inline?
function isclique(G)
    return 2*binomial(nv(G), 2) == ne(G)
end

"""
    fac(n, fmemo)

Recursively compute the factorial of n using memoization table fmemo.

"""
function fac(n, fmemo)
    fmemo[n] != 0 && return fmemo[n]
    n == 1 && return BigInt(1)
    res = fac(n-1, fmemo) * n
    return fmemo[n] = res
end

function rho(X, memo)
    haskey(memo.rho, X) && return memo.rho[X]
    res = fac(first(X), memo.fac)
    for j = 2:length(X) # TODO: use eachindex?
        res -= fac(first(X) - X[j], memo.fac) * rho(view(X, j:length(X)), memo)
    end
    return memo.rho[X] = res
end

"""
    cliquetreefrommcs(G, mcsorder, invmcsorder)

Compute a clique tree of graph G based on a previously compute mcsorder (and its inverse). Implements Alg. 4 from "Computing a Clique Tree with the Algorithm Maximal Label Search" by Berry and Simonet (https://www.mdpi.com/1999-4893/10/1/20).

"""
function cliquetreefrommcs(G, mcsorder, invmcsorder)
    n = nv(G)
    # data structures for the algorithm
    K = Vector{Set}()
    push!(K, Set())
    s = 1
    edgelist = Set{Graphs.SimpleGraphs.SimpleEdge}()
    visited = falses(n)
    clique = zeros(Int, n)

    for i = 1:n
        x = mcsorder[i]
        S = Set{Int}()
        for w in inneighbors(G, x)
            if visited[w]
                push!(S, w)
            end
        end
        
        # if necessary create new maximal clique
        if K[s] != S
            s += 1
            push!(K, S)
            # TODO: findmax is rather slow I noticed
            k, _ = findmax(map(x -> invmcsorder[x], collect(S)))
            p = clique[mcsorder[k]]
            push!(edgelist, Graphs.SimpleGraphs.SimpleEdge(p, s))
        end
        
        union!(K[s], x)
        clique[x] = s 
        visited[x] = true;
    end

    T = SimpleGraphFromIterator(edgelist)
    # ensure graph is not empty
    nv(T) == 0 && add_vertices!(T,1)
    return K, T
end

"""
    mcs(G, K)

Performs a Maximum Cardinality Search on graph G. The elements of K are of prioritised and chosen first. Returns the visit order of the vertices, its inverse and the subgraphs C_G(K) (see Def. 1 in [1,2]). If K is empty a normal MCS is performed.

The MCS takes the role of the LBFS in [1,2]. Details will be published in future work.

"""
function mcs(G)
    n = nv(G)
    
    # data structures for MCS
    sets = [LinkedList{Int}() for _ = 1:n+1]
    pointers = Vector(undef,n)
    size = ones(Int64, n)
    
    # output data structures
    mcsorder = Vector{Int}(undef, n)
    invmcsorder = Vector{Int}(undef, n)

    # init
    for v in vertices(G)
        pointers[v] = push!(sets[1], v)
    end
    maxcard = 1

    for i = 1:n
        v = first(sets[maxcard])
        # v is the ith vertex in the mcsorder
        mcsorder[i] = v
        invmcsorder[v] = i
        size[v] = -1

        deleteat!(sets[maxcard], pointers[v])

        # update the neighbors
        for w in inneighbors(G, v)
            if size[w] >= 1
                deleteat!(sets[size[w]], pointers[w])
                size[w] += 1
                pointers[w] = push!(sets[size[w]], w)
            end
        end
        maxcard += 1
        while maxcard >= 1 && isempty(sets[maxcard])
            maxcard -= 1
        end
    end

    return mcsorder, invmcsorder
end

"""
    cliquetree(G)

Computes a clique tree of a graph G. A vector K of maximal cliques and a tree T on 1,2,...,|K| is returned.

"""
function cliquetree(G)
    mcsorder, invmcsorder = mcs(G)
    K, T = cliquetreefrommcs(G, mcsorder, invmcsorder)
    return K, T
end


function count(K, T, E, Et, F, X, S, R, P, memo)
    if P > length(E)
        #println("id: " * string(P))
        #println("init")
    else 
        s, t = src(E[R[P]]), dst(E[R[P]])
        #println("id: " * string(P))
        #println(string(K[s]) * " " * string(K[t]))
    end
    P = R[P]
    memo.amos[P] != 0 && return memo.amos[P]
    length(F[P]) == 1 && return memo.amos[P] = fac(length(K[first(F[P])]) - S[P], memo.fac)
    sum = BigInt(0)
    # for clique in subproblem
    for i in F[P] 
        #println("hi" * string(i))
        # compute phi
        Xi = [length(K[i]) - S[P]]
        for (e, l) in X[i]
            ed = E[e]
            s, t = src(ed), dst(ed)
            if !(s in F[P]) || !(t in F[P])
                break 
            end
            l - S[P] > 0 && push!(Xi, l - S[P]) 
        end
        #println(Xi)
        phi = rho(Xi, memo) 
        #println(phi)
        #println("??")

        # compute subproblems
        prod = BigInt(1)
        q = [i]
        vis = Set{Int64}()
        out = Set{Int64}()
        push!(vis, i)
        push!(out, i)
        while !isempty(q)
            u = pop!(q)
            for v in neighbors(T, u)
                !(v in F[P]) && continue
                if !(v in vis)
                    push!(q, v)
                    push!(vis, v)
                end
                if !(v in out)
                    prod *= count(K, T, E, Et, F, X, S, R, Et[Edge(u, v)], memo)
                    union!(out, F[R[Et[Edge(u, v)]]])
                end
            end
        end
        # combine
        sum += phi * prod
    end

    #println("id: " * string(P))
    #println(sum)
    return memo.amos[P] = sum
end

function find_flower(K, T, e, R, Et)
    s, t = src(e), dst(e)
    S = intersect(K[s], K[t])
    F = Set{Int64}()
    push!(F, s)
    push!(F, t)
    q = [t]
    while !isempty(q)
        u = pop!(q)
        for v in neighbors(T, u)
            if !(v in F) && issubset(S, K[v])
                if intersect(K[u], K[v]) == S 
                    # TODO: -> this edge points to same flower!
                    R[Et[Edge(v, u)]] = Et[e]
                    continue
                end
                push!(F, v)
                push!(q, v)
            end
        end
    end
    delete!(F, s)
    return F
end

function find_X(K, T, Et)
    # X contains a vector of edge, sz tuples for each clique
    # indicates separators in phi and their sizes
    X = [Vector{Tuple{Int64, Int64}}() for _ = 1:nv(T)]
    vis = falses(nv(T))
    pred = -1 * ones(Int64, nv(T))
    q = [1]
    vis[1] = true
    while !isempty(q)
        u = pop!(q)
        for v in neighbors(T, u)
            if !vis[v]
                push!(q, v)
                vis[v] = true
                pred[v] = u
            end
        end
        idx = u
        while pred[idx] != -1 
            S = intersect(K[idx], K[pred[idx]])
            if (isempty(X[u]) || last(X[u])[2] > length(S)) && issubset(S, K[u])
                push!(X[u], (Et[Edge(idx, pred[idx])], length(S)))
            end
            idx = pred[idx]
        end
    end
    return X
end

function countamos(G, memo)
    isclique(G) && return fac(nv(G), memo.fac)
    #x = @elapsed begin
    K, T = cliquetree(G)
    #println("K: " * string(K))
    #println("T: " * string(T))
    # maybe do this preprocessing in function
    E = Vector{Edge}()
    for e in edges(T)
        push!(E, e)
        push!(E, reverse(e))
    end
    Et = Dict{Edge, Int64}()
    for i in eachindex(E)
        Et[E[i]] = i
    end
    F = Vector{Set{Int64}}()
    R = -1 * ones(Int64, length(E))
    for i in eachindex(E)
        if R[i] == -1 
            R[i] = i
            push!(F, find_flower(K, T, E[i], R, Et))
        else 
            push!(F, Set{Int64}())
        end
    end
    push!(R, length(R)+1)
    # this is initial graph, which corresponds to no edge in clique tree
    push!(F, Set{Int64}(vertices(T)))
    X = find_X(K, T, Et)
    S = Vector{Int64}()
    # is also computed in find_X btw
    for i in eachindex(E)
        i % 2 == 0 && continue
        nw = length(intersect(K[src(E[i])], K[dst(E[i])]))
        push!(S, nw)
        push!(S, nw)
    end
    push!(S, 0)
    #end
    #println(x)
    return count(K, T, E, Et, F, X, S, R, length(F), memo)
end

"""
    MECsize(G)

Return the number of Markov equivalent DAGs in the class represented by CPDAG G.

# Examples
```julia-repl
julia> G = readgraph("example.in", true)
{6, 22} directed simple Int64 graph
julia> MECsize(G)
54
```
"""
function MECsize(G, withchecks = true)
    n = nv(G)
    memo = Memo(zeros(BigInt, 2*n), Dict{Vector, BigInt}(), zeros(BigInt, n)) 
    U = copy(G)
    U.ne = 0
    for i = 1:n
        filter!(j->has_edge(G, j, i), U.fadjlist[i])
        filter!(j->has_edge(G, i, j), U.badjlist[i])
        U.ne += length(U.fadjlist[i])
    end
    tres = 1
    for component in connected_components(U)
        H, _ = induced_subgraph(U, component) # TODO: how slow is induced_subgraph?
        if withchecks && !ischordal(H)
            println("Undirected connected components are NOT chordal...Abort")
            println("Are you sure the graph is a CPDAG?")
            # is there anything more clever than just returning?
            return
        end
        tres *= countamos(H, memo)
    end

    return tres
end
