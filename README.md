# Clique-Picking

This repository provides functionality for working with CPDAGs, which are graphs representing Markov equivalence classes (MEC) of DAGs common in causal discovery. Currently, it focuses mostly on algorithms for counting, sampling and listing the DAGs in the MEC represented by a given CPDAG based on [[3]](https://epub.uni-luebeck.de/items/44cd6c2b-c86a-40bc-aef2-669310429ac6). 

The algorithms are implemented in Rust and exposed via an Python wrapper available as PyPI package ```cliquepicking```. An R wrapper is planned.

## Installation

The python package can be installed with:
```
pip install cliquepicking
```

For more detailed documentation, see the README in the ```cliquepicking_python``` folder. 

## Roadmap

The goal is to implement further algorithms from my thesis, such as those for MPDAGs. Feature requests are welcome. 

## Prototypes
The ```prototypes/``` directory contains code I originally wrote in the context of the papers [[1]](https://arxiv.org/abs/2012.09679) and [[2]](https://www.jmlr.org/papers/v24/22-0495.html), which is not maintained anymore. 

## References
1. Marcel Wienöbst, Max Bannach, and Maciej Liśkiewicz: *Polynomial-Time Algorithms for Counting and Sampling Markov Equivalent DAGs* (AAAI 2021) [arXiv version](https://arxiv.org/abs/2012.09679)
2. Marcel Wienöbst, Max Bannach, and Maciej Liśkiewicz: *Polynomial-Time Algorithms for Counting and Sampling Markov Equivalent DAGs with Applications* [(JMLR)](https://www.jmlr.org/papers/v24/22-0495.html)
3. Marcel Wienöbst: *Algorithms for Markov Equivalence* [(PhD Thesis)](https://epub.uni-luebeck.de/items/44cd6c2b-c86a-40bc-aef2-669310429ac6)
