import cliquepicking as cp

# DISCLAIMER: I currently use this for some basic profiling
# file reading is hacky

filename = "../prototypes/aaai_experiments/instances/peo-4/peo-n=4096-4-nr=1.gr"
with open(filename, "r") as file:
    next(file)
    next(file)
    input = [tuple(map(int, line.split())) for line in file]
    cpdag = []
    for pair in input:
        cpdag.append((pair[0], pair[1]))
        cpdag.append((pair[1], pair[0]))
    print(cp.mec_size(cpdag))
    sampler = cp.MecSampler(cpdag)
    print(len(sampler.sample_dag()))
