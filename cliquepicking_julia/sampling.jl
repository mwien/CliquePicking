using LightGraphs
using Random

include("counting.jl")
include("utils.jl")

"""
Struct storing the precomputation information.
"""
mutable struct Memo
    alias::Tuple{Vector{BigInt},Vector{Int}, BigInt} # alias table allowing to draw in O(1)
    clique::Vector{Vector} # mapping index -> clique
    FP::Vector{Vector} # mapping index -> FP
    SP::Vector{Vector} # mapping index -> subproblems
    Memo(a,b,c) = (x = new(); x.clique = a; x.FP = b; x.SP = c; x)
end

"""
    alias_initialization(weights, s)

Initialize the alias table used for efficiently drawing a clique. The array weights contains the weights of the respective elements, while s is their sum.
"""
function alias_initialization(weights, s)
    weights_copy = copy(weights)
    n = length(weights_copy)
    alias = zeros(BigInt, n)
    prob = zeros(BigInt, n)

    small = []
    large = []

    for i=1:n
        weights_copy[i] *= n
        if weights_copy[i] < s
            push!(small, i)
        else
            push!(large, i)
        end
    end

    while !isempty(small) && !isempty(large)
        l = pop!(small)
        g = pop!(large)

        prob[l] = weights_copy[l]
        alias[l] = g
        weights_copy[g] = (weights_copy[g] + weights_copy[l]) - s
        if weights_copy[g] < s
            push!(small, g)
        else
            push!(large, g)
        end
    end

    for g in large
        prob[g] = s
    end

    return prob, alias, s
end

"""
    drawalias((prob, alias, s))

Draw an element from the alias table given through prob and alias with s being the sum of all weights. Note that prob is scaled by s, which avoids floating point operations.
"""
function drawalias((prob, alias, s))
    i = rand(1:length(prob))
    x = rand(1:s)
    return x <= prob[i] ? i : alias[i] 
end

"""
    precomputation(G)

Perform necessary precomputations for graph G to allow for fast sampling of a DAG.

# Examples
```julia-repl
julia> G = readgraph("example.in", true)
{6, 22} directed simple Int64 graph
julia> precomputation(G)
(Memo[Memo((BigInt[54, 36, 36], [0, 1, 2], 54), Array{T,1} where T[[2, 3, 5, 6], [4, 2, 3, 5], [2, 3, 1]], Array{T,1} where T[Any[], Any[[2, 3, 5]], Any[[2, 3]]], Array{T,1} where T[[2, 3], [5, 3], [4]]), Memo((BigInt[1], [0], 1), Array{T,1} where T[[4]], Array{T,1} where T[Any[]], Array{T,1} where T[Int64[]]), Memo((BigInt[1], [0], 1), Array{T,1} where T[[1]], Array{T,1} where T[Any[]], Array{T,1} where T[Int64[]]), Memo((BigInt[3, 2], [0, 1], 3), Array{T,1} where T[[5, 4], [5, 6]], Array{T,1} where T[Any[], Any[[5]]], Array{T,1} where T[[5], [2]]), Memo((BigInt[1], [0], 1), Array{T,1} where T[[6]], Array{T,1} where T[Any[]], Array{T,1} where T[Int64[]])], [1])
```
"""
function precomputation(G)
    memoID = Dict{Set, Tuple{BigInt, Int}}() # mapping set of vertices to #AMO and index
    memo = Vector{Memo}()
    startids = Vector{Int}()
    fmemo = zeros(BigInt, nv(G))

    U = copy(G)
    U.ne = 0
    for i = 1:nv(G)
        filter!(j->has_edge(G, j, i), U.fadjlist[i])
        filter!(j->has_edge(G, i, j), U.badjlist[i])
        U.ne += length(U.fadjlist[i])
    end
    for component in connected_components(U)
        cc = induced_subgraph(G, component)
        if !ischordal(cc[1])
            println("Undirected connected components are NOT chordal...Abort")
            println("Are you sure the graph is a CPDAG?")
            # is there anything more clever than just returning? maybe throw an error?
            return
        end
        push!(startids, length(memoID)+1)
        countwithprecomputation(cc, memoID, memo, fmemo)
    end
    return memo, startids
end

"""
    fptopos!(pos, FPsets, K)

Map vertices of K to the size of the FP element where it first occurs and saves it in pos. If it never occurs, it has the value of |K|+1. Preprocessing in order to draw permutations of cliques fast.
"""
function fptopos!(pos, FPsets, K)
    for i in K
        pos[i] = length(K) + 1
    end

    for i = 1:length(FPsets)
        for j in FPsets[i]
            pos[j] = length(FPsets[i])
        end
    end
end

"""
    isallowed(perm, FP)

Checks if a permutation perm starts with a forbidden prefix from FP.
"""
function isallowed(perm, FP)
    mx = 0
    for i = 1:length(perm)
        mx = max(mx, FP[perm[i]])
        mx == i && return false
        mx > length(perm) && return true
    end
    return true
end

"""
    drawpermutation(K, FP)

Draw an allowed permutation of the clique K that is not part of the forbidden prefixes in FP.

Instead of the Alg. 4 from [2], we use simple rejection sampling for this subtask. Details will be published in future work.
"""
function drawpermutation(K, FP)
    perm = shuffle(K)

    while !isallowed(perm, FP)
        perm = shuffle(K)
    end

    return perm
end

"""
    sampleAMO!(time, tick, pos, idG, memo)

Sample a uniform AMO from graph identified with idG recursively. The AMO is represented through the visit order of the vertices stored in time. comp is used for labeling the vertices with the id of the UCCG compid, tick is a counter for the visit times, pos is used as storage when drawing clique permutations, and memo contains all necessary precomputation information.

"""
function sampleAMO!(comp, time, tick, pos, idG, memo, compid)
    # draw a clique K
    cliqueidx = drawalias(memo[idG].alias)
    K = memo[idG].clique[cliqueidx]

    fptopos!(pos, memo[idG].FP[cliqueidx], K)
    # sample permutation
    perm = drawpermutation(K, pos)

    for i in perm
        comp[i] = compid
        time[i] = tick
        tick = tick + 1
    end
    
    # recurse into subproblems
    for idH in memo[idG].SP[cliqueidx]
        tick = sampleAMO!(comp, time, tick, pos, idH, memo, compid)
    end
    
    return tick
end

"""
    sampleDAG(G)

Return a sampled DAG from for a chordal graph G.

If multiple DAGs are sampled from the same graph G, it is recommended to use sampleDAG(G, pre).

# Examples
```julia-repl
julia> G = readgraph("example.in", true)
{6, 22} directed simple Int64 graph
julia> sampleDAG(G)
{6, 11} directed simple Int64 graph
```
"""
function sampleDAG(G)
    pre = precomputation(G)
    return sampleDAG(G, pre)
end

"""
    sampleDAG(G, pre)

Return a sampled DAG for a chordal graph G using already computed precomputation pre.

# Examples
```julia-repl
julia> G = readgraph("example.in", true)
{6, 22} directed simple Int64 graph
julia> pre = precomputation(G);
julia> sampleDAG(G, pre)
{6, 11} directed simple Int64 graph
```
"""
function sampleDAG(G, pre)
    n = nv(G)
    memo = pre[1]
    startids = pre[2]

    comp = zeros(Int, n)
    compid = 1
    time = zeros(Int, n)
    tick = 1

    pos = zeros(Int, n)
    
    for idG in startids
        tick = sampleAMO!(comp, time, tick, pos, idG, memo, compid)
        compid += 1
    end

    D = copy(G)
    D.ne = 0
    for i = 1:n
        filter!(j->comp[i] != comp[j] || time[j] > time[i], D.fadjlist[i])
        filter!(j->comp[i] != comp[j] || time[j] < time[i], D.badjlist[i])
        D.ne += length(D.fadjlist[i])
    end

    return D
end
