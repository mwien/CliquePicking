using LightGraphs

function readGraph()
    (n, m) = parse.(Int, split(readline()))
    readline()
    g = SimpleDiGraph(n)
    for i = 1:m
        (a, b) = parse.(Int, split(readline()))
        add_edge!(g, a, b)
        add_edge!(g, b, a)
    end

    return (n, g)
end

function computeCliquePicking(g)
    gg = SimpleGraphFromIterator(edges(intersect(g, reverse(g))))

    CPs = connected_components(gg)

    tres = 1
    for comp in CPs
        cc = induced_subgraph(gg,comp)[1]
        open("wrapper.in", "w") do io
            write(io, string(nv(cc), " ", ne(cc), "\n\n"))
            for i in edges(cc)
                write(io, string(src(i), " ", dst(i), "\n"))
            end
        end;
        tmp = read(pipeline("wrapper.in", `./a.out`), String)
        tres *= parse(Int, tmp)
    end

    rm("wrapper.in")

    return tres
end

###### TEST ######
(n, g) = readGraph()
result = computeCliquePicking(g)
println(result)