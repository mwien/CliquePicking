include("counting.jl")
include("oldcounting.jl")
include("generate.jl")

#function experiments(verbose = 0)
#   for (root, _, files) in walkdir("instances/")
#        for file in files
#            println("Working on " * file)
#            G = readgraph(joinpath(root, file), true)
#            if nv(G) > 2000
#               continue
#            end
#            newtime = @elapsed newres = MECsize(G)
#            GC.gc()
#            GC.gc()
#            GC.gc()
#            GC.gc()
#            oldtime = @elapsed oldres = MECsize(G, oldcount)
#            GC.gc()
#            GC.gc()
#            GC.gc()
#            GC.gc()
#            @assert newres == oldres
#            println(string(newtime) * " " * string(oldtime))
#        end
#    end
#end

function infchecker(verbose = 0)
   s = 1000
   while true 
      n = rand(50:100)
      m = rand(n:10*n)
      G = gencc(n, m, s)
      newres = MECsize(G)
      GC.gc()
      GC.gc()
      GC.gc()
      GC.gc()
      oldres = MECsize(G, oldcount)
      GC.gc()
      GC.gc()
      GC.gc()
      GC.gc()
      if newres != oldres 
         println(s)
         println(string(newres) * " " * string(oldres))
         println(G)
         return G
      end
      s += 1
      sleep(0.1)
      s % 100 == 0 && println("still alive " * string(s))
   end
end

function checker(verbose = 0)
   s = 1000
   while true 
      n = 6
      m = rand(6:9)
      G = gencc(n, m, s)
      newres = MECsize(G)
      GC.gc()
      GC.gc()
      GC.gc()
      GC.gc()
      oldres = oldMECsize(G, oldcount)
      GC.gc()
      GC.gc()
      GC.gc()
      GC.gc()
      if newres != oldres 
         println(s)
         println(string(newres) * " " * string(oldres))
         println(G)
         return G
      end
      s += 1
      #sleep(0.1)
      #s % 100 == 0 && println("still alive " * string(s))
   end
end
