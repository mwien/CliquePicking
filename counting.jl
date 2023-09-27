using Graphs
using LinkedLists

include("utils.jl")

# TODO: inline?
function isclique(G)
    return binomial(nv(G), 2) == ne(G)
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

function rho(X, rmemo, fmemo)
    haskey(rmemo, X) && return rmemo[X]
    res = fac(first(X), fmemo)
    for j = 2:length(X) # TODO: use eachindex?
        res -= fac(first(X) - X[j], fmemo) * rho(view(X, j:length(X)), rmemo, fmemo)
    end
    return res
end

"""
    phi(cliquesize, i, fp, fmemo, pmemo)

Recursively compute the function phi. Initial call is phi(cliquesize, 1, fp, fmemo, pmemo). If not used before, fmemo and pmemo should be zero vectors of appropriate size. 

"""
function phi(cliquesize, i, fp, fmemo, pmemo)
    pmemo[i] != 0 && return pmemo[i]
    sum = fac(cliquesize-fp[i], fmemo)
    for j = (i+1):length(fp)
        sum -= fac(fp[j]-fp[i], fmemo) * phi(cliquesize, j, fp, fmemo, pmemo)
    end
    return pmemo[i] = sum
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

function find_flower(K, T, s, t)
    S = intersect(K[s], K[t])
    F = Set{Int64}()
    push!(F, s)
    push!(F, t)
    q = [t]
    while !isempty(q)
        u = pop!(q)
        for v in neighbors(T, u)
            if !(v in F) && issubset(S, K[v])
                push!(F, v)
                push!(q, v)
            end
        end
    end
    delete!(F, s)
    return F
end

# TODO: very bad name
function find_rho(K, T, Et)
    X = [Vector{Tuple{Int64, Int64}}() for _ = 1:nv(T)]
    q = [1]
    vis = falses(nv(T))
    pred = -1 * ones(Int64, nv(T))
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
            if (isempty(X[u]) || last(X)[2] > length(S)) && issubset(S, K[u])
                push!(X[u], (Et[Edge(idx, pred[idx])], length(S)))
            end
        end
    end
    return X
end

function count(K, T, E, Et, F, X, S, P, memo)
    # doesn't uniquely ID subproblem!
    # let's not worry about this now
    memo[P] != 0 && return memo[P]
    sum = 0
    for i in F[P] 
        # compute phi
        Xi = [length(K[i])]
        for (e, l) in X[i]
            ed = Et[e]
            s = src(ed)
            t = dst(ed)
            if !(s in F[P]) && !(t in F[P])
                break # is this correct?
            end
            push!(Xi, l - S[P])
        end
        phi = rho(Xi, memo[2], memo[3]) 

        # compute subproblems
        prod = 1
        q = [i]
        visited = falses(nv(T)) # TODO: could also reuse this to save further memory
        # TODO: yeah its bad cause we don't want to always have effort O(n)! -> falses(length(F[P]))
        output = falses(nv(T))
        pred = -1 * ones(Int64, nv(T))
        visited[i] = true # TODO: forgot this at other places
        output[i] = true
        while !isempty(q)
            u = pop!(q)
            for v in neighbors(T, u)
                !(v in F[P]) && continue
                if !visited[v]
                    push!(q, v)
                    visited[v] = true
                    pred[v] = u
                end
                if !output[v]
                    # recursive call for subproblem (u, v) edge
                    # then mark output all in FP[(u,v)]
                end
            end
        end

        # combine
        sum += phi * prod
    end
    return memo[P] = sum
end

function countamos(G, memo)
    isclique(G) && return fac(nv(G), memo[3])
    K, T = cliquetree(G)
    E = Vector{Edge}()
    for e in edges(T)
        push!(E, e)
        push!(E, reverse(e))
    end
    # Edge or just tuple maybe
    Et = Dict{Edge, Int64}()
    for i in eachindex(E)
        Et[E[i]] = i
    end
    F = Vector{Set{Int64}}()
    for i in eachindex(E)
        F[i] = find_flower(K, T, src(E[i]), dst(E[i]))
    end
    push!(F, Set{Int64}(vertices(T)))
    X = find_rho(K, T, Et)
    S = Vector{Int64}()
    # is also computed in find_rho btw
    for i in eachindex(E)
        i % 2 == 0 && continue
        nw = length(intersect(K[src(E[i])], K[dst(E[i])]))
        push!(S, nw)
        push!(S, nw)
    end
    return count(K, T, E, Et, F, X, S, length(F), memo)

#    # check memoization table
#    mapG = Set(map(x -> mapping[x], vertices(G)))
#    haskey(memo, mapG) && return memo[mapG]
#    
#    # do bfs over the clique tree
#    K, T = cliquetree(G)
#    sum = BigInt(0)
#    Q = [1]
#    vis = falses(nv(T))
#    vis[1] = true
#    pred = -1 * ones(Int, nv(T))
#    while !isempty(Q)
#        v = pop!(Q)
#        for x in inneighbors(T, v)
#            if !vis[x]
#                push!(Q, x)
#                vis[x] = true
#                pred[x] = v
#            end
#        end
#
#        # product of #AMOs for the subproblems
#        prod = BigInt(1)
#        for H in subproblems(G, K[v])
#            HH = induced_subgraph(G, H)
#            prod *= count((HH[1], map(x -> mapping[x], HH[2])), memo, fmemo)
#        end
#
#        # compute correction term phi
#        FP = []
#        curr = v
#        curr_succ = -1
#        intersect_pred = -1
#        while pred[curr] != -1
#            curr = pred[curr]
#            intersect_v = length(intersect(K[v], K[curr]))
#            if curr_succ != -1
#                intersect_pred = length(intersect(K[curr], K[curr_succ]))
#            end
#            curr_succ = curr
#            if intersect_v == 0
#                break
#            end
#            #if lastcut were strictly greater, v is not in bouquet
#            # defined by cut between curr and curr_succ
#            if intersect_v >= intersect_pred && (isempty(FP) || intersect_v < FP[end])
#                push!(FP, intersect_v)
#            end
#        end
#        push!(FP, 0)
#        pmemo = zeros(BigInt, length(FP))
#        sum += prod * phi(length(K[v]), 1, reverse(FP), fmemo, pmemo)
#    end
#    return memo[mapG] = sum
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
    memo = (zeros(BigInt, 2*n), Dict{Vector, BigInt}(), zeros(BigInt, n)) # could also do memo struct
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
        tres *= count(H, memo)
    end

    return tres
end

"""
    countwithprecomputation(component, memoID, memo, fmemo)

Main counting function with additional precomputations needed for fast sampling.
memoID contains the ID for the given component and memo and fmemo allow memoization.
"""
function countwithprecomputation(component, memoID, memo, fmemo)
    G = component[1] # graph
    mapping = component[2] # mapping to original vertex numbers
    n = nv(G)

    # check memoization table
    mapG = Set(map(x -> mapping[x], vertices(G)))
    haskey(memoID, mapG) && return memoID[mapG]
    
    # do bfs over the clique tree
    idG = length(memo)+1 # id for mapG
    K, T = cliquetree(G)
    push!(memo, Memo(Vector(undef, length(K)), Vector(undef, length(K)), Vector(undef, length(K))))
    amosperclique = Vector{BigInt}(undef, length(K))
    sum = 0
    Q = [1]
    FP = []
    vis = falses(nv(T))
    vis[1] = true
    pred = -1 * ones(Int, nv(T))
    while !isempty(Q)
        v = pop!(Q)
        memo[idG].clique[v] = map(x -> mapping[x], collect(K[v]))
        memo[idG].FP[v] = []
        memo[idG].SP[v] = Vector{Int}()

        for x in inneighbors(T, v)
            if !vis[x]
                push!(Q, x)
                vis[x] = true
                pred[x] = v
            end
        end

        # product of #AMOs for the subproblems
        prod = BigInt(1)
        for H in subproblems(G, K[v]) 
            HH = induced_subgraph(G, H)
            res, id = countwithprecomputation((HH[1], map(x -> mapping[x], HH[2])), memoID, memo, fmemo)
            prod *= res
            push!(memo[idG].SP[v], id)
        end
        
        # compute correction term phi
        FP = []
        FPsets = []
        curr = v
        curr_succ = -1
        intersect_pred = -1
        while pred[curr] != -1
            curr = pred[curr]
            intersect_v = length(intersect(K[v], K[curr]))
            if curr_succ != -1
                intersect_pred = length(intersect(K[curr], K[curr_succ]))
            end
            curr_succ = curr
            if intersect_v == 0
                break
            end
            # if lastcut were strictly greater, v is not in bouquet
            # defined by cut between curr and curr_succ
            if intersect_v >= intersect_pred && (isempty(FP) || intersect_v < FP[end])
                push!(FP, intersect_v)
                push!(FPsets, intersect(K[v], K[curr]))
            end
        end

        tmp = Set()
        for x in reverse(FPsets)
            if isempty(memo[idG].FP[v])
                push!(memo[idG].FP[v], map(y -> mapping[y], collect(x)))
            else
                push!(memo[idG].FP[v], map(y -> mapping[y], collect(setdiff(x, tmp))))
            end
            tmp = x
        end
        
        push!(FP, 0)
        pmemo = zeros(BigInt, length(FP))
        sum += amosperclique[v] = prod * phi(length(K[v]), 1, reverse(FP), fmemo, pmemo)
    end
    # init alias
    memo[idG].alias = alias_initialization(amosperclique, sum)
    memoID[mapG] = (sum, idG) # assign ID
    return memoID[mapG]
end
