using Graphs
using LinkedLists

include("utils.jl")

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
function oldcount(cc, memo, fmemo)
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
            prod *= oldcount((HH[1], map(x -> mapping[x], HH[2])), memo, fmemo)
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

function oldMECsize(G, count_fun = oldcount)
    n = nv(G)
    memo = Dict{Set, BigInt}() #mapping set of vertices -> AMO sum
    fmemo = zeros(BigInt, n)
    U = copy(G)
    U.ne = 0
    for i = 1:n
        filter!(j->has_edge(G, j, i), U.fadjlist[i])
        filter!(j->has_edge(G, i, j), U.badjlist[i])
        U.ne += length(U.fadjlist[i])
    end
    tres = 1
    for component in connected_components(U)
        cc = induced_subgraph(U, component)
        if !ischordal(cc[1])
            println("Undirected connected components are NOT chordal...Abort")
            println("Are you sure the graph is a CPDAG?")
            # is there anything more clever than just returning?
            return
        end
        tres *= count_fun(cc, memo, fmemo)
    end

    #println(length(memo))

    return tres
end
