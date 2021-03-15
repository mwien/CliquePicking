using LightGraphs
using LinkedLists

include("utils.jl")

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
    edgelist = Set{LightGraphs.SimpleGraphs.SimpleEdge}()
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
            k, _ = findmax(map(x -> invmcsorder[x], collect(S)))
            p = clique[mcsorder[k]]
            push!(edgelist, LightGraphs.SimpleGraphs.SimpleEdge(p, s))
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
    vispush!(l::LinkedList, pointers, x, vis)

"""
@inline function vispush!(l::LinkedList, pointers, x, vis)
    if vis
        pointers[x] = push!(l,x)
    else
        pointers[x] = pushfirst!(l,x)
    end
end

"""
    mcs(G, K)

Performs a Maximum Cardinality Search on graph G. The elements of K are of prioritised and chosen first. Returns the visit order of the vertices, its inverse and the subgraphs C_G(K) (see Def. 1 in [1,2]). If K is empty a normal MCS is performed.

The MCS takes the role of the LBFS in [1,2]. Details will be published in future work.

"""
function mcs(G, K)
    n = nv(G)
    copy_K = copy(K)
    
    # data structures for MCS
    sets = [LinkedList{Int}() for i = 1:n+1]
    pointers = Vector(undef,n)
    size = Vector{Int}(undef, n)
    visited = falses(n)
    
    # output data structures
    mcsorder = Vector{Int}(undef, n)
    invmcsorder = Vector{Int}(undef, n)
    subgraphs = Array[]

    # init
    visited[collect(copy_K)] .= true
    for v in vertices(G)
        size[v] = 1
        vispush!(sets[1], pointers, v, visited[v])
    end
    maxcard = 1

    for i = 1:n
        # first, the vertices in K are chosen
        # they are always in the set of maximum cardinality vertices
        if !isempty(copy_K)
            v = pop!(copy_K)
        # afterwards, the algorithm chooses any vertex from maxcard
        else
            v = first(sets[maxcard])
        end
        # v is the ith vertex in the mcsorder
        mcsorder[i] = v
        invmcsorder[v] = i
        size[v] = -1

        # immediately append possible subproblems to the output
        if !visited[v]
            vertexset = Vector{Int}()
            for x in sets[maxcard]
                visited[x] && break
                visited[x] = true
                push!(vertexset, x)
            end
            sg = induced_subgraph(G, vertexset)
            subgraphs = vcat(subgraphs, (map(x -> sg[2][x], connected_components(sg[1]))))
        end

        deleteat!(sets[maxcard], pointers[v])

        # update the neighbors
        for w in inneighbors(G, v)
            if size[w] >= 1
                deleteat!(sets[size[w]], pointers[w])
                size[w] += 1
                vispush!(sets[size[w]], pointers, w, visited[w])
            end
        end
        maxcard += 1
        while maxcard >= 1 && isempty(sets[maxcard])
            maxcard -= 1
        end
    end

    return mcsorder, invmcsorder, subgraphs
end

"""
    cliquetree(G)

Computes a clique tree of a graph G. A vector K of maximal cliques and a tree T on 1,2,...,|K| is returned.

"""
function cliquetree(G)
    mcsorder, invmcsorder, _ = mcs(G, Set())
    K, T = cliquetreefrommcs(G, mcsorder, invmcsorder)
    return K, T
end

"""
    subproblems(G, K)

Computes C_G(K) (see Def. 1).

"""
function subproblems(G, K)
    _, _, subgraphs = mcs(G, K)
    return subgraphs
end

"""
    count(cc, memo, fmemo)

Main counting function. Implements the recursion given in Prop. 3 of [1,2]. The graph is given as the pair cc, describing the graph itself as well as the mapping of the vertives to its original vertex numbers.
memo as well as fmemo are necessary for memoization, memo stores previously handled subgraphs, fmemo is used for the precomputation of factorials.
"""
function count(cc, memo, fmemo)
    G = cc[1] # graph
    mapping = cc[2] # mapping to original vertex numbers
    n = nv(G)
    
    # check memoization table
    mapG = Set(map(x -> mapping[x], vertices(G)))
    haskey(memo, mapG) && return memo[mapG]
    
    # do bfs over the clique tree
    K, T = cliquetree(G)
    sum = BigInt(0)
    Q = [1]
    vis = falses(nv(T))
    vis[1] = true
    pred = -1 * ones(Int, nv(T))
    while !isempty(Q)
        v = pop!(Q)
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
            prod *= count((HH[1], map(x -> mapping[x], HH[2])), memo, fmemo)
        end

        # compute correction term phi
        FP = []
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
            #if lastcut were strictly greater, v is not in bouquet
            # defined by cut between curr and curr_succ
            if intersect_v >= intersect_pred && (isempty(FP) || intersect_v < FP[end])
                push!(FP, intersect_v)
            end
        end
        push!(FP, 0)
        pmemo = zeros(BigInt, length(FP))
        sum += prod * phi(length(K[v]), 1, reverse(FP), fmemo, pmemo)
    end
    return memo[mapG] = sum
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
function MECsize(G)
    n = nv(G)
    memo = Dict{Set, BigInt}() #mapping set of vertices -> AMO sum
    fmemo = zeros(BigInt, n)
    U = SimpleGraphFromIterator(edges(intersect(G, reverse(G))))

    tres = 1
    for component in connected_components(U)
        cc = induced_subgraph(U, component)
        if !ischordal(cc[1])
            println("Undirected connected components are NOT chordal...Abort")
            println("Are you sure the graph is a CPDAG?")
            # is there anything more clever than just returning?
            return
        end
        tres *= count(cc, memo, fmemo)
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
