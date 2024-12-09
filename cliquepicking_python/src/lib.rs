use pyo3::prelude::*;

use cliquepicking_rs::count::count_cpdag;
use cliquepicking_rs::partially_directed_graph::PartiallyDirectedGraph;
use cliquepicking_rs::sample::sample_cpdag;
use cliquepicking_rs::sample::sample_cpdag_orders;

use num_bigint::BigUint;

/// A Python module for counting and sampling Markov equivalent DAGs.
#[pymodule]
fn cliquepicking(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(mec_size, m)?)?;
    m.add_function(wrap_pyfunction!(mec_sample_dags, m)?)?;
    m.add_function(wrap_pyfunction!(mec_sample_orders, m)?)?;
    Ok(())
}

/// Compute the number of DAGs in the Markov equivalence represented by CPDAG cpdag.
#[pyfunction]
fn mec_size(cpdag: Vec<(usize, usize)>) -> PyResult<BigUint> {
    let mx = max_element(&cpdag);
    let g = PartiallyDirectedGraph::from_edge_list(cpdag, mx + 1);
    let res = count_cpdag(&g);
    Ok(res)
}

/// Sample k DAGs uniformly from the Markov equivalence class represented by CPDAG cpdag.
#[pyfunction]
fn mec_sample_dags(cpdag: Vec<(usize, usize)>, k: usize) -> PyResult<Vec<Vec<(usize, usize)>>> {
    let mx = max_element(&cpdag);
    let g = PartiallyDirectedGraph::from_edge_list(cpdag, mx + 1);
    let samples = sample_cpdag(&g, k)
        .into_iter()
        .map(|sample| sample.to_edge_list())
        .collect();
    Ok(samples)
}

/// Sample k DAGs uniformly from the Markov equivalence class represented by CPDAG cpdag.
#[pyfunction]
fn mec_sample_orders(cpdag: Vec<(usize, usize)>, k: usize) -> PyResult<Vec<Vec<usize>>> {
    let mx = max_element(&cpdag);
    let g = PartiallyDirectedGraph::from_edge_list(cpdag, mx + 1);
    Ok(sample_cpdag_orders(&g, k))
}

// small helper
fn max_element(tuple_list: &[(usize, usize)]) -> usize {
    let mut mx = 0;
    for &(u, v) in tuple_list.iter() {
        mx = std::cmp::max(u, mx);
        mx = std::cmp::max(v, mx);
    }
    mx
}
