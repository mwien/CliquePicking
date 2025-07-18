use crate::index_set::IndexSet;
use crate::partially_directed_graph::PartiallyDirectedGraph;
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

    /// Finds the CPDAG (aka essential graph) of a given DAG. This methods implements the
    /// algorithm given by Chickering (1995) in a way such that it runs in worst-case linear time
    /// (the original algorithm has expected linear run-time due to relying on hash tables for
    /// adjacency checks).
    pub fn to_cpdag(&self) -> PartiallyDirectedGraph {
        let mut edge_list = Vec::new();

        let top_order = self.topological_order();
        // parents ordered according to top_order
        let mut parents = vec![Vec::new(); self.n];
        for &u in top_order.iter() {
            for &v in self.outneighbors(u) {
                parents[v].push(u);
            }
        }

        // let closest parent of v be C(v), then we check for each v
        // whether there is a v-structure C(v) -> v <- z for some z
        // closest_in_vs stores this info
        let mut closest_in_vs = vec![false; self.n];
        // child_in_flag is re-usable memory for lookup of children
        let mut child_flag = vec![usize::MAX; self.n];

        for z in 0..self.n {
            for &y in self.outneighbors(z) {
                child_flag[y] = z;
            }

            for &y in self.outneighbors(z) {
                // x is closest parent of y in top_order
                if let Some(&x) = parents[y].last() {
                    // in the future: use let chains to merge the ifs
                    if x != z && child_flag[x] != z {
                        closest_in_vs[y] = true;
                    }
                }
            }
        }

        // tracks compelled parents
        let mut compelled_parents = vec![Vec::new(); self.n];
        // re-usable memory for lookup of compelled parents
        let mut compelled_flag = vec![usize::MAX; self.n];
        // re-usable memory for lookup of parents
        let mut parent_flag = vec![usize::MAX; self.n];

        for y in top_order {
            if parents[y].is_empty() {
                continue;
            }
            // x is closest parent of y in top_order
            let x = *parents[y].last().unwrap();
            if closest_in_vs[y] {
                parents[y].iter().for_each(|&u| compelled_flag[u] = y);
            } else {
                parents[y].iter().for_each(|&w| parent_flag[w] = y);
                for &w in compelled_parents[x].iter() {
                    if parent_flag[w] != y {
                        parents[y].iter().for_each(|&u| compelled_flag[u] = y);
                        break;
                    } else {
                        compelled_flag[w] = y;
                    }
                }
            }
            for &u in parents[y].iter() {
                if compelled_flag[u] == y {
                    compelled_parents[y].push(u);
                    edge_list.push((u, y));
                } else {
                    edge_list.push((u, y));
                    edge_list.push((y, u));
                }
            }
        }
        PartiallyDirectedGraph::from_edge_list(edge_list, self.n)
    }
}
