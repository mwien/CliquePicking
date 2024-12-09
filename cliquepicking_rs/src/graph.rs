use crate::directed_graph::DirectedGraph;
use crate::index_set::IndexSet;
use crate::utils::{edge_list_from_stdin, inverse_permutation};
use std::collections::VecDeque;

#[derive(Clone, Debug)]
pub struct Graph {
    pub n: usize,
    pub m: usize,
    neighbors: Vec<IndexSet>,
}

// undirected graph
// here it is assumed that each edge is given only once in stdin or edge list when creating a Graph
impl Graph {
    pub fn from_adjacency_list(adjacency_list: Vec<Vec<usize>>) -> Graph {
        let n = adjacency_list.len();
        let m = adjacency_list.iter().map(|x| x.len()).sum::<usize>() / 2;
        let neighbors = adjacency_list.into_iter().map(IndexSet::from).collect();

        Graph { n, m, neighbors }
    }

    pub fn from_edge_list(edge_list: Vec<(usize, usize)>, n: usize) -> Graph {
        let mut adjacency_list = vec![Vec::new(); n];
        for &(u, v) in &edge_list {
            adjacency_list[u].push(v);
            adjacency_list[v].push(u);
        }
        Graph::from_adjacency_list(adjacency_list)
    }

    pub fn from_stdin() -> Graph {
        let (edge_list, n) = edge_list_from_stdin();
        Graph::from_edge_list(edge_list, n)
    }

    pub fn from_index_sets(index_sets: Vec<IndexSet>) -> Graph {
        let n = index_sets.len();
        let m = index_sets.iter().map(|x| x.len()).sum::<usize>() / 2;

        Graph {
            n,
            m,
            neighbors: index_sets,
        }
    }

    pub fn neighbors(&self, u: usize) -> std::slice::Iter<'_, usize> {
        self.neighbors[u].iter()
    }

    pub fn bfs_ordering(&self) -> Vec<usize> {
        let mut queue = VecDeque::new();
        let mut visited = vec![false; self.n];
        let mut visit_ordering = Vec::new();

        queue.push_back(0);
        visited[0] = true;

        while !queue.is_empty() {
            let u = queue.pop_front().unwrap();
            visit_ordering.push(u);
            for &v in self.neighbors(u) {
                if !visited[v] {
                    queue.push_back(v);
                    visited[v] = true;
                }
            }
        }
        visit_ordering
    }

    pub fn connected_components(&self) -> (Vec<Graph>, Vec<Vec<usize>>) {
        let mut queue = VecDeque::new();
        let mut visited = vec![usize::MAX; self.n]; // rename
        let mut new_id = vec![usize::MAX; self.n];
        let mut cnt = 0;

        let mut component_vertices = Vec::new();
        for i in 0..self.n {
            if visited[i] == usize::MAX {
                let mut component = Vec::new();
                queue.push_back(i);
                visited[i] = cnt;
                new_id[i] = component.len();
                component.push(i);
                while let Some(u) = queue.pop_front() {
                    for &v in self.neighbors(u) {
                        if visited[v] == usize::MAX {
                            queue.push_back(v);
                            visited[v] = cnt;
                            new_id[v] = component.len();
                            component.push(v);
                        }
                    }
                }
                component_vertices.push(component);
                cnt += 1;
            }
        }
        let mut adjacency_lists = Vec::new();
        for component in &component_vertices {
            adjacency_lists.push(vec![Vec::new(); component.len()]);
        }
        for i in 0..self.n {
            for &j in self.neighbors(i) {
                adjacency_lists[visited[i]][new_id[i]].push(new_id[j]);
            }
        }
        (
            adjacency_lists
                .into_iter()
                .map(Graph::from_adjacency_list)
                .collect(),
            component_vertices,
        )
    }

    pub fn orient(&self, ordering: &[usize]) -> DirectedGraph {
        let inv_ordering = inverse_permutation(ordering);
        let mut adjacency_list = vec![Vec::new(); self.n];

        for i in 0..self.n {
            for &j in self.neighbors(i) {
                if inv_ordering[j] > inv_ordering[i] {
                    adjacency_list[i].push(j);
                }
            }
        }

        DirectedGraph::from_adjacency_list(adjacency_list)
    }
}

#[cfg(test)]
mod tests {
    use super::Graph;

    #[test]
    fn connected_components_basic_check() {
        let g = Graph::from_edge_list(vec![(0, 1), (4, 2), (4, 5)], 6);
        let (_, comps) = g.connected_components();
        assert_eq!(comps.len(), 3);
        // TODO: more tests
        // maybe have helper functions -> set equality (recursive)
    }
}
