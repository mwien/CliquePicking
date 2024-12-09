use crate::index_set::IndexSet;
use crate::utils;

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct DirectedGraph {
    pub n: usize,
    outneighbors: Vec<IndexSet>,
}

impl DirectedGraph {
    pub fn from_adjacency_list(adjacency_list: Vec<Vec<usize>>) -> DirectedGraph {
        let n = adjacency_list.len();
        let outneighbors = adjacency_list
            .clone()
            .into_iter()
            .map(IndexSet::from)
            .collect();
        DirectedGraph { n, outneighbors }
    }

    pub fn from_edge_list(edge_list: Vec<(usize, usize)>, n: usize) -> DirectedGraph {
        let mut adjacency_list = vec![Vec::new(); n];
        for &(u, v) in &edge_list {
            adjacency_list[u].push(v);
        }
        DirectedGraph::from_adjacency_list(adjacency_list)
    }

    pub fn from_stdin() -> DirectedGraph {
        let (edge_list, n) = utils::edge_list_from_stdin();
        DirectedGraph::from_edge_list(edge_list, n)
    }

    pub fn from_outneighbors(outneighbors: Vec<IndexSet>) -> DirectedGraph {
        let n = outneighbors.len();
        DirectedGraph { n, outneighbors }
    }

    pub fn outneighbors(&self, u: usize) -> std::slice::Iter<'_, usize> {
        self.outneighbors[u].iter()
    }

    pub fn to_adjacency_list(&self) -> Vec<Vec<usize>> {
        self.outneighbors
            .clone()
            .into_iter()
            .map(|x| x.to_vec())
            .collect()
    }

    pub fn to_edge_list(&self) -> Vec<(usize, usize)> {
        let mut edge_list = Vec::new();
        for u in 0..self.n {
            for &v in self.outneighbors(u) {
                edge_list.push((u, v));
            }
        }
        edge_list
    }

    pub fn topological_order_dfs(&self, vis: &mut Vec<bool>, ord: &mut Vec<usize>, u: usize) {
        if vis[u] {
            return;
        }
        vis[u] = true;
        for &v in self.outneighbors(u) {
            self.topological_order_dfs(vis, ord, v);
        }
        ord.push(u);
    }

    pub fn topological_order(&self) -> Vec<usize> {
        let mut vis = vec![false; self.n];
        let mut ord: Vec<usize> = Vec::new();
        for u in 0..self.n {
            if !vis[u] {
                self.topological_order_dfs(&mut vis, &mut ord, u);
            }
        }
        ord.reverse();
        ord
    }
}
