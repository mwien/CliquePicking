from typing import List, Tuple

def mec_size(cpdag: List[Tuple[int, int]]) -> int:
    """
    Compute the number of DAGs in the Markov equivalence class represented by the CPDAG.

    Parameters
    -----------
    cpdag: A list of tuples representing the edges of the CPDAG, where each tuple is a pair of integers (vertex1, vertex2). Undirected edges are encoded by including both (a, b) and (b, a).

    Returns
    -------
    An integer representing the number of DAGs.
    """
    ...

def mec_list_dags(cpdag: List[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
    """
    List all DAGs in the Markov equivalence class represented by the CPDAG.

    Parameters
    -----------
    cpdag: A list of tuples representing the edges of the CPDAG, where each tuple is a pair of integers (vertex1, vertex2). Undirected edges are encoded by including both (a, b) and (b, a).

    Returns
    -------
    A list of DAGs, where each DAG is represented as a list of tuples (edges).
    """
    ...

def mec_list_orders(cpdag: List[Tuple[int, int]]) -> List[List[int]]:
    """
    List all DAGs (represented by a topological order) in the Markov equivalence class represented by the CPDAG.

    Parameters
    -----------
    cpdag: A list of tuples representing the edges of the CPDAG, where each tuple is a pair of integers (vertex1, vertex2). Undirected edges are encoded by including both (a, b) and (b, a).

    Returns
    -------
    A list of topological orders, where each topological order is represented as a list of integers (vertex indices).
    """
    ...

class MecSampler:
    def __init__(self, cpdag: List[Tuple[int, int]]) -> None: ...
    def sample_dag(self) -> List[Tuple[int, int]]: ...
    def sample_order(self) -> List[int]: ...
