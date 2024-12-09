use crate::directed_graph::DirectedGraph;
use crate::graph::Graph;
use crate::index_set::IndexSet;
use crate::utils;

// an edge x-y is interpreted as undirected if x ∈ inneighbors[y] and x ∈ outneighbors[y]
// note that x ∈ inneighbors[y] implies y ∈ outneighbors[x]
#[derive(Clone, Debug)]
pub struct PartiallyDirectedGraph {
    pub n: usize,
    inneighbors: Vec<IndexSet>,
    outneighbors: Vec<IndexSet>,
}

// partially directed graph
// here it is assumed that an undirected edge is given as two directed edges in stdin or edge list
// when creating a PartiallyDirectedGraph
impl PartiallyDirectedGraph {
    pub fn from_adjacency_list(adjacency_list: Vec<Vec<usize>>) -> PartiallyDirectedGraph {
        let n = adjacency_list.len();
        let outneighbors = adjacency_list
            .clone()
            .into_iter()
            .map(IndexSet::from)
            .collect();

        let mut transposed_adjacency_list = vec![Vec::new(); adjacency_list.len()];
        for (i, adjacencies) in adjacency_list.iter().enumerate() {
            for &j in adjacencies {
                transposed_adjacency_list[j].push(i);
            }
        }
        let inneighbors = transposed_adjacency_list
            .into_iter()
            .map(IndexSet::from)
            .collect();

        PartiallyDirectedGraph {
            n,
            inneighbors,
            outneighbors,
        }
    }

    pub fn from_edge_list(edge_list: Vec<(usize, usize)>, n: usize) -> PartiallyDirectedGraph {
        let mut adjacency_list = vec![Vec::new(); n];
        for &(u, v) in &edge_list {
            adjacency_list[u].push(v);
        }
        PartiallyDirectedGraph::from_adjacency_list(adjacency_list)
    }

    pub fn from_stdin() -> PartiallyDirectedGraph {
        let (edge_list, n) = utils::edge_list_from_stdin();
        PartiallyDirectedGraph::from_edge_list(edge_list, n)
    }

    pub fn undirected_subgraph(&self) -> Graph {
        let mut index_sets = Vec::new();
        for i in 0..self.n {
            index_sets.push(self.inneighbors[i].intersection(&self.outneighbors[i]));
        }
        Graph::from_index_sets(index_sets)
    }

    pub fn directed_subgraph(&self) -> DirectedGraph {
        let mut outneighbors = Vec::new();
        for i in 0..self.n {
            outneighbors.push(self.outneighbors[i].set_difference(&self.inneighbors[i]));
        }
        DirectedGraph::from_outneighbors(outneighbors)
    }
}
