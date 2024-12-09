use std::collections::VecDeque;

use crate::chordal;
use crate::graph::Graph;
use crate::index_set::IndexSet;
use crate::utils;

// TODO: think about storing flowers and separators here as well
#[derive(Clone, Debug)]
pub struct CliqueTree {
    pub cliques: Vec<IndexSet>,
    pub tree: Graph,
    rank: Vec<usize>, // used internally to assign edge ids
    dists: Vec<usize>,
}

impl CliqueTree {
    pub fn from(g: &Graph) -> CliqueTree {
        let mcs_ordering = chordal::mcs(g);
        let inv_mcs_ordering = utils::inverse_permutation(&mcs_ordering);
        // maybe check chordality here and return option

        let mut cliques = Vec::new();
        let mut current_clique = Vec::new();
        let mut tree_edges = Vec::new();

        let mut num_cliques = 0;
        let mut visited = vec![false; g.n]; // bitset maybe
        let mut clique_id = vec![0; g.n]; // maybe rename

        for &u in &mcs_ordering {
            let visited_neighbors =
                IndexSet::from_sorted(g.neighbors(u).cloned().filter(|&v| visited[v]).collect());
            if !visited_neighbors.equal_to_vec(&current_clique) {
                num_cliques += 1;
                let k = visited_neighbors
                    .iter()
                    .copied()
                    .max_by_key(|&x| inv_mcs_ordering[x])
                    .unwrap(); // unwrap is safe if g is connected
                let p = clique_id[k];
                tree_edges.push((p, num_cliques));
                cliques.push(IndexSet::from(current_clique));
                current_clique = visited_neighbors.to_vec();
            }
            current_clique.push(u);
            clique_id[u] = num_cliques;
            visited[u] = true;
        }

        // assume non-empty graph
        cliques.push(IndexSet::from(current_clique));
        let tree = Graph::from_edge_list(tree_edges, cliques.len());

        // compute ranks
        let bfs_ordering = tree.bfs_ordering();
        let rank = utils::inverse_permutation(&bfs_ordering);

        // compute dists
        visited = vec![false; tree.n];
        let mut queue = VecDeque::new();
        queue.push_back(0);
        visited[0] = true;

        let mut dists = vec![0; tree.n];
        dists[0] = 0;

        while !queue.is_empty() {
            let u = queue.pop_front().unwrap();
            for &v in tree.neighbors(u) {
                if !visited[v] {
                    visited[v] = true;
                    dists[v] = dists[u] + 1;
                    queue.push_back(v);
                }
            }
        }

        CliqueTree {
            cliques,
            tree,
            rank,
            dists,
        }
    }

    pub fn get_edge_id(&self, u: usize, v: usize) -> usize {
        if self.rank[u] < self.rank[v] {
            2 * (self.rank[v] - 1) + 1
        } else {
            2 * (self.rank[u] - 1)
        }
    }

    pub fn separators(&self) -> Vec<IndexSet> {
        let mut separators = vec![IndexSet::new(); 2 * (self.tree.n - 1)]; // could also
                                                                           // start with empty vec and then add separators (need to iterator over edges in id
                                                                           // order) -> maybe edge list is useful later anyway
        for u in 0..self.tree.n {
            // maybe Graph needs an iterator over adjacency lists
            for &v in self.tree.neighbors(u) {
                separators[self.get_edge_id(u, v)] = self.cliques[u].intersection(&self.cliques[v]);
                // TODO: speedup possible here by not computing everything twice
            }
        }
        separators
    }

    // for now just copy flower in case same one occurs multiple times
    // could have more efficient storage with pointers to flowers
    // same holds for separators
    pub fn flowers(&self, separators: &[IndexSet]) -> Vec<IndexSet> {
        let mut flowers = vec![IndexSet::new(); 2 * (self.tree.n - 1)];
        let mut visited = vec![false; self.tree.n];

        for s in 0..self.tree.n {
            // as before
            for &t in self.tree.neighbors(s) {
                let edge_id = self.get_edge_id(s, t);
                let st_sep = &separators[edge_id];
                if flowers[edge_id].is_empty() {
                    let mut flower = Vec::new();
                    flower.push(t);
                    let mut add_ids = Vec::new();
                    add_ids.push(edge_id);
                    visited[s] = true;
                    visited[t] = true;
                    let mut q = VecDeque::new();
                    q.push_back(t);
                    while !q.is_empty() {
                        let u = q.pop_front().unwrap();
                        for &v in self.tree.neighbors(u) {
                            if !visited[v] && st_sep.is_subset(&self.cliques[v]) {
                                if separators[self.get_edge_id(u, v)] == *st_sep {
                                    add_ids.push(self.get_edge_id(u, v));
                                } else {
                                    flower.push(v);
                                    visited[v] = true;
                                    q.push_back(v);
                                }
                            }
                        }
                    }
                    visited[s] = false;
                    for &f in &flower {
                        visited[f] = false;
                    }
                    for &id in &add_ids {
                        flowers[id] = IndexSet::from(flower.clone());
                    }
                }
            }
        }
        flowers
    }

    pub fn forbidden_sets(
        &self,
        separators: &[IndexSet],
        flowers: &[IndexSet],
    ) -> Vec<Vec<(usize, usize, usize)>> {
        let mut forbidden_sets: Vec<Vec<(usize, usize, usize)>> = vec![Vec::new(); self.tree.n];

        for u in 0..self.tree.n {
            for &v in self.tree.neighbors(u) {
                if self.dists[u] > self.dists[v] {
                    continue;
                }
                let edge_id = self.get_edge_id(u, v);
                for &clique_id in &flowers[edge_id] {
                    forbidden_sets[clique_id].push((u, v, separators[edge_id].len()));
                }
            }
        }

        for forbidden_set in forbidden_sets.iter_mut() {
            forbidden_set.sort_by_key(|x| x.2);
            forbidden_set.reverse();
        }

        forbidden_sets
    }
}

// TODO: write tests? -> check properties of clique tree
#[cfg(test)]
mod tests {
    use super::CliqueTree;
    use crate::graph::Graph;

    #[test]
    fn from_basic_check() {
        let g =
            Graph::from_adjacency_list(vec![vec![1, 2, 3], vec![0, 3], vec![0, 3], vec![0, 1, 2]]);
        let clique_tree = CliqueTree::from(&g);
        assert_eq!(clique_tree.tree.n, 2);
        // TODO: better tests
    }

    #[test]
    fn separators_basic_check() {
        let g =
            Graph::from_adjacency_list(vec![vec![1, 2, 3], vec![0, 3], vec![0, 3], vec![0, 1, 2]]);
        let clique_tree = CliqueTree::from(&g);
        let separators = clique_tree.separators();
        assert_eq!(separators.len(), 2);
        // TODO: better tests
    }

    #[test]
    fn flowers_basic_check() {
        let g =
            Graph::from_adjacency_list(vec![vec![1, 2, 3], vec![0, 3], vec![0, 3], vec![0, 1, 2]]);
        let clique_tree = CliqueTree::from(&g);
        let separators = clique_tree.separators();
        let flowers = clique_tree.flowers(&separators);
        assert_eq!(flowers.len(), 2);
        // TODO: better tests
    }
}
