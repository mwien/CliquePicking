# Clique-Picking

This repository provides functionality for working with CPDAGs, which are graphs representing Markov equivalence classes (MEC) of DAGs common in causal discovery. Currently, it focuses mostly on algorithms for counting, sampling and listing the DAGs in the MEC represented by a given CPDAG based on my [PhD thesis](https://mwien.github.io/thesis.pdf). 

The algorithms are implemented in Rust and exposed via an Python wrapper available as Pypi package ```cliquepicking```. An R wrapper is planned.

## Installation

The python package can be installed with
```
pip install cliquepicking
```

For more detailed documentation, see the README in the ```cliquepicking_python``` folder. 

## Roadmap

The goal is to implement further algorithms from [3], such as those for MPDAGs. Feature requests are welcome. 

## References
The repository contains an implementation of the Clique-Picking algorithm proposed at AAAI 2021 [1] for counting the  number of Markov equivalent DAGs in polynomial time. 
Moreover, it provides implementations of the polynomial-time algorithm for uniformly sampling a DAG from a Markov Equivalence Class, which were refined in [2]. See also Chapter 5 of [3]. 

1. Marcel Wienöbst, Max Bannach, and Maciej Liśkiewicz: *Polynomial-Time Algorithms for Counting and Sampling Markov Equivalent DAGs* (AAAI 2021) [arXiv version](https://arxiv.org/abs/2012.09679)
2. Marcel Wienöbst, Max Bannach, and Maciej Liśkiewicz: *Polynomial-Time Algorithms for Counting and Sampling Markov Equivalent DAGs with Applications* [(JMLR)](https://www.jmlr.org/papers/v24/22-0495.html)
3. Marcel Wienöbst: *Algorithms for Markov Equivalence* [(PhD Thesis)](https://mwien.github.io/thesis.pdf)

## Prototypes
The prototypes directory contains code I wrote in the context of the papers [1] and [2], which is not maintained anymore. 
