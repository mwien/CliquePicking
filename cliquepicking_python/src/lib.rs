use cliquepicking_rs::sample::CpdagSampler;
use pyo3::prelude::*;

use cliquepicking_rs::count::count_cpdag;
use cliquepicking_rs::enumerate::list_cpdag;
use cliquepicking_rs::enumerate::list_cpdag_orders;
use cliquepicking_rs::partially_directed_graph::PartiallyDirectedGraph;

use num_bigint::BigUint;

/// A Python module for counting and sampling Markov equivalent DAGs.
#[pymodule]
fn cliquepicking(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(mec_size, m)?)?;
    m.add_class::<MecSampler>()?;
    m.add_function(wrap_pyfunction!(mec_list_dags, m)?)?;
    m.add_function(wrap_pyfunction!(mec_list_orders, m)?)?;
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

#[pyclass]
struct MecSampler(CpdagSampler);

#[pymethods]
impl MecSampler {
    #[new]
    fn new(cpdag: Vec<(usize, usize)>) -> Self {
        let mx = max_element(&cpdag);
        let g = PartiallyDirectedGraph::from_edge_list(cpdag, mx + 1);
        MecSampler(CpdagSampler::init(&g))
    }

    fn sample_dag(&self) -> Vec<(usize, usize)> {
        self.0.sample_dag(&mut rand::thread_rng()).to_edge_list()
    }

    fn sample_order(&self) -> Vec<usize> {
        self.0.sample_order(&mut rand::thread_rng())
    }
}

/// List all DAGs from the Markov equivalence class represented by CPDAG cpdag.
#[pyfunction]
fn mec_list_dags(cpdag: Vec<(usize, usize)>) -> PyResult<Vec<Vec<(usize, usize)>>> {
    let mx = max_element(&cpdag);
    let g = PartiallyDirectedGraph::from_edge_list(cpdag, mx + 1);
    let samples = list_cpdag(&g)
        .into_iter()
        .map(|sample| sample.to_edge_list())
        .collect();
    Ok(samples)
}

/// List all DAGs (represented by a topological orderfrom the Markov equivalence class represented by CPDAG cpdag.
#[pyfunction]
fn mec_list_orders(cpdag: Vec<(usize, usize)>) -> PyResult<Vec<Vec<usize>>> {
    let mx = max_element(&cpdag);
    let g = PartiallyDirectedGraph::from_edge_list(cpdag, mx + 1);
    Ok(list_cpdag_orders(&g))
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
