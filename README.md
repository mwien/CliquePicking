# The Clique-Picking Algorithm

This repository countains multiple implementations of the Clique-Picking algorithm that we have proposed at AAAI 2021 [1] for counting the  number of Markov equivalent DAGs in polynomial time. 
Moreover, we give implementations of the polynomial-time algorithm for uniformly sampling a DAG from a Markov Equivalence Class, which we refined in [2]. See also chapter 5 of my thesis [3]. 

1. Marcel Wienöbst, Max Bannach, and Maciej Liśkiewicz: *Polynomial-Time Algorithms for Counting and Sampling Markov Equivalent DAGs* (AAAI 2021) [arXiv version](https://arxiv.org/abs/2012.09679)
2. Marcel Wienöbst, Max Bannach, and Maciej Liśkiewicz: *Polynomial-Time Algorithms for Counting and Sampling Markov Equivalent DAGs with Applications* [(JMLR)](https://www.jmlr.org/papers/v24/22-0495.html)
3. Marcel Wienöbst: *Algorithms for Markov Equivalence* [Thesis](https://mwien.github.io/thesis.pdf)

## Implementations

We provide the following implementations:

- ```aaai_experiments/``` contains the original C++ implementation and experiments from the AAAI 2021 conference paper (only counting is implemented here)
- ```cliquepicking_julia/``` contains the subsequent Julia code additionally provides a sampling algorithm
- ```cliquepicking_rs/``` contains a recent Rust implementation for counting and sampling, which is cleaner and slightly more efficient (worst-case n^3)
- ```cliquepicking_python/``` contains a Python wrapper for the Rust implementation. You can install and use the Python implementation easily via ```pip install cliquepicking```.

Brief examples of how to install and use the implementations are given in the subdirectories.
