use num_bigint::{BigUint, RandBigInt};
use num_traits::One;
use rand::rngs::ThreadRng;
use rand::seq::SliceRandom;
use rand::Rng;
use std::cmp;
use std::collections::VecDeque;

use crate::clique_tree::CliqueTree;
use crate::combinatorics;
use crate::directed_graph::DirectedGraph;
use crate::graph::Graph;
use crate::index_set::IndexSet;
use crate::lazy_tokens::LazyTokens;
use crate::memoization::Memoization;
use crate::partially_directed_graph::PartiallyDirectedGraph;

#[derive(Debug, Clone)]
struct AliasTable {
    n: usize,
    prob: Vec<BigUint>, // maybe call this weight
    alias: Vec<usize>,
    total: BigUint,
}

impl AliasTable {
    // sample would panic on this
    fn new_empty() -> Self {
        AliasTable {
            n: 0,
            prob: Vec::new(),
            alias: Vec::new(),
            total: BigUint::ZERO,
        }
    }
    fn init(weights: &[BigUint], total: BigUint) -> Self {
        let n = weights.len();
        let mut weights_cloned = weights.to_vec();
        let mut alias = vec![0; n];
        let mut prob = vec![BigUint::ZERO; n];

        let mut small = Vec::new();
        let mut large = Vec::new();

        weights_cloned
            .iter_mut()
            .enumerate()
            .for_each(|(i, weight)| {
                *weight *= n;
                if *weight < total {
                    small.push(i);
                } else {
                    large.push(i);
                }
            });

        while !small.is_empty() && !large.is_empty() {
            let l = small.pop().unwrap();
            let g = large.pop().unwrap();

            prob[l] = weights_cloned[l].clone();
            alias[l] = g;
            weights_cloned[g] =
                weights_cloned[g].clone() + weights_cloned[l].clone() - total.clone();
            if weights_cloned[g] < total {
                small.push(g);
            } else {
                large.push(g);
            }
        }

        for &g in &large {
            prob[g] = total.clone();
        }

        Self {
            n,
            prob,
            alias,
            total,
        }
    }

    fn sample(&self, rng: &mut ThreadRng) -> usize {
        let i = rng.gen_range(0..self.n);
        let x = rng.gen_biguint_below(&self.total);
        if x < self.prob[i] {
            i
        } else {
            self.alias[i]
        }
    }
}

#[derive(Debug)]
struct Sampler {
    n: usize,
    clique_tree: CliqueTree,
    separators: Vec<IndexSet>, // integrate separators and flowers into clique_tree
    flowers: Vec<IndexSet>,
    alias_tables: Vec<AliasTable>,
    forbidden_sets: Vec<Vec<(usize, usize, usize)>>,
}

impl Sampler {
    fn init(g: &Graph) -> Self {
        // compute the clique tree of G
        let clique_tree = CliqueTree::from(g);
        let mut separators = clique_tree.separators();
        separators.push(IndexSet::from_sorted(Vec::new()));

        let mut flowers = clique_tree.flowers(&separators);
        flowers.push(IndexSet::from_sorted((0..clique_tree.tree.n).collect()));

        let num_subproblems = flowers.len();

        let mut alias_tables = vec![AliasTable::new_empty(); num_subproblems];
        let mut forbidden_sets = clique_tree.forbidden_sets(&separators, &flowers);

        let mut visited = LazyTokens::new(clique_tree.tree.n);
        let mut considered = LazyTokens::new(clique_tree.tree.n);
        let mut memoization = Memoization::new(clique_tree.tree.n, g.n);

        // init alias_tables and forbidden_sets
        Self::rec_count_init(
            &mut alias_tables,
            &mut forbidden_sets,
            num_subproblems - 1,
            &mut visited,
            &mut considered,
            &mut memoization,
            &clique_tree,
            &separators,
            &flowers,
        );

        Self {
            n: g.n,
            clique_tree,
            separators,
            flowers,
            alias_tables,
            forbidden_sets,
        }
    }

    fn rec_count_init(
        alias_tables: &mut [AliasTable],
        forbidden_sets: &mut [Vec<(usize, usize, usize)>],
        subproblem: usize,
        visited: &mut LazyTokens,
        considered: &mut LazyTokens,
        memoization: &mut Memoization,
        clique_tree: &CliqueTree,
        separators: &[IndexSet],
        flowers: &[IndexSet],
    ) -> BigUint {
        if memoization.count[subproblem] != BigUint::ZERO {
            // already computed
            return memoization.count[subproblem].clone(); // clone?
        }
        let flower = &flowers[subproblem];
        let separator = &separators[subproblem];
        let mut sum = BigUint::ZERO;
        let mut amos_per_clique = vec![BigUint::ZERO; flower.len()];
        // -> quadratic cost are prob. not bottleneck i guess
        let mut pre = vec![BigUint::ZERO; 2 * (clique_tree.tree.n - 1)];
        for (i, &clique_id) in flower.iter().enumerate() {
            let mut forbidden_sizes = Vec::new();
            forbidden_sizes.push(clique_tree.cliques[clique_id].len() - separator.len());
            for &(u, v, size) in &forbidden_sets[clique_id] {
                if !flower.contains(u) || !flower.contains(v) {
                    break;
                }
                if size > separator.len() {
                    forbidden_sizes.push(size - separator.len());
                } else {
                    break; // should be correct
                }
            }
            let phi = combinatorics::rho(&forbidden_sizes, memoization);

            visited.prepare();
            considered.prepare();
            visited.set(clique_id);
            considered.set(clique_id);

            let product = phi
                * Self::rec_count_traversal(
                    alias_tables,
                    forbidden_sets,
                    flower,
                    clique_id,
                    &mut pre,
                    visited,
                    considered,
                    memoization,
                    clique_tree,
                    separators,
                    flowers,
                );

            visited.restore();
            considered.restore();

            sum += product.clone();
            amos_per_clique[i] = product;
        }
        memoization.count[subproblem].clone_from(&sum);
        alias_tables[subproblem] = AliasTable::init(&amos_per_clique, sum.clone()); // could also
                                                                                    // pass reference of sum
        sum
    }

    fn rec_count_traversal(
        alias_tables: &mut [AliasTable],
        forbidden_sets: &mut [Vec<(usize, usize, usize)>],
        flower: &IndexSet,
        clique_id: usize,
        pre: &mut Vec<BigUint>,
        visited: &mut LazyTokens,
        considered: &mut LazyTokens,
        memoization: &mut Memoization,
        clique_tree: &CliqueTree,
        separators: &[IndexSet],
        flowers: &[IndexSet],
    ) -> BigUint {
        visited.set(clique_id);
        let mut product = BigUint::one();
        for &next_clique_id in clique_tree.tree.neighbors(clique_id) {
            if !flower.contains(next_clique_id) {
                continue;
            }
            let edge_id = clique_tree.get_edge_id(clique_id, next_clique_id);
            if !visited.check(next_clique_id) && !considered.check(next_clique_id) {
                if pre[edge_id] != BigUint::ZERO {
                    visited.set(next_clique_id);
                    product *= pre[edge_id].clone();
                } else {
                    let next_flower_result = Self::rec_count_init(
                        alias_tables,
                        forbidden_sets,
                        edge_id,
                        visited,
                        considered,
                        memoization,
                        clique_tree,
                        separators,
                        flowers,
                    );
                    for &new_clique_id in &flowers[edge_id] {
                        considered.set(new_clique_id);
                    }
                    let remaining_subtree_result = Self::rec_count_traversal(
                        alias_tables,
                        forbidden_sets,
                        flower,
                        next_clique_id,
                        pre,
                        visited,
                        considered,
                        memoization,
                        clique_tree,
                        separators,
                        flowers,
                    );
                    pre[edge_id] = next_flower_result * remaining_subtree_result;
                    product *= pre[edge_id].clone();
                }
            } else if !visited.check(next_clique_id) {
                product *= Self::rec_count_traversal(
                    alias_tables,
                    forbidden_sets,
                    flower,
                    next_clique_id,
                    pre,
                    visited,
                    considered,
                    memoization,
                    clique_tree,
                    separators,
                    flowers,
                );
            }
        }
        product
    }

    fn sample_ordering(&self, rng: &mut ThreadRng) -> Vec<usize> {
        self.rec_sample_ordering(
            self.flowers.len() - 1,
            &mut vec![0; self.n],
            &mut LazyTokens::new(self.clique_tree.tree.n),
            &mut LazyTokens::new(self.clique_tree.tree.n),
            rng,
        )
    }

    fn rec_sample_ordering(
        &self,
        subproblem: usize,
        pos: &mut Vec<usize>,
        visited: &mut LazyTokens,
        considered: &mut LazyTokens,
        rng: &mut ThreadRng,
    ) -> Vec<usize> {
        let mut ordering = Vec::new();

        let clique_id = self.flowers[subproblem].get(self.alias_tables[subproblem].sample(rng));
        let clique = &self.clique_tree.cliques[clique_id];
        let flower = &self.flowers[subproblem];
        let separator = &self.separators[subproblem];

        let remaining_clique_vertices = clique.set_difference(separator);
        let mut forbidden_prefixes = Vec::new();
        for &(u, v, size) in &self.forbidden_sets[clique_id] {
            if !flower.contains(u) || !flower.contains(v) {
                break;
            }
            if size > separator.len() {
                forbidden_prefixes.push(
                    self.separators[self.clique_tree.get_edge_id(u, v)].set_difference(separator),
                );
            } else {
                break;
            }
        }
        let mut clique_ordering =
            Self::draw_allowed_permutation(&remaining_clique_vertices, pos, &forbidden_prefixes);
        ordering.append(&mut clique_ordering);

        // could also pre-compute subproblems however that would increase
        // the run-time of counting for sparse graphs to n^3
        // precomputation would likely only improve run-time for finding ordering from O(n+m) to O(n)
        let flower = &self.flowers[subproblem];
        let mut queue = VecDeque::new();
        queue.push_back(clique_id);
        visited.prepare();
        considered.prepare();
        visited.set(clique_id);
        considered.set(clique_id);
        while !queue.is_empty() {
            let u = queue.pop_front().unwrap();
            for &v in self.clique_tree.tree.neighbors(u) {
                if !flower.contains(v) {
                    continue;
                }
                if !visited.check(v) {
                    queue.push_back(v);
                    visited.set(v);
                }
                if !considered.check(v) {
                    let new_flower_id = self.clique_tree.get_edge_id(u, v);
                    ordering.append(&mut self.rec_sample_ordering(
                        new_flower_id,
                        pos,
                        visited,
                        considered,
                        rng,
                    ));
                    for &new_clique_id in &self.flowers[new_flower_id] {
                        considered.set(new_clique_id);
                    }
                }
            }
        }
        visited.restore();
        considered.restore();
        ordering
    }

    fn draw_allowed_permutation(
        clique: &IndexSet,
        helper: &mut [usize],
        forbidden_prefixes: &Vec<IndexSet>,
    ) -> Vec<usize> {
        // first extract smallest set a vertex in clique appears in
        for &u in clique {
            helper[u] = clique.len();
        }
        // overwriting should be correct as forbidden prefixes are ordered from biggest to smallest
        for forbidden_prefix in forbidden_prefixes {
            for &u in forbidden_prefix {
                helper[u] = forbidden_prefix.len() - 1;
            }
        }

        let mut rng = rand::thread_rng();

        loop {
            let mut perm = clique.clone().to_vec();
            perm.shuffle(&mut rng);
            if Self::is_allowed(&perm, helper) {
                return perm;
            }
        }
    }

    fn is_allowed(perm: &[usize], helper: &[usize]) -> bool {
        let mut mx = 0;
        for (i, &u) in perm.iter().enumerate() {
            mx = cmp::max(mx, helper[u]);
            if mx == i {
                return false;
            }
            if mx >= perm.len() {
                return true;
            }
        }
        true
    }
}

pub struct CpdagSampler {
    samplers: Vec<Sampler>,
    ordered_comps: Vec<Vec<usize>>,
    undirected_subgraph: Graph,
    directed_subgraph: DirectedGraph,
}

impl CpdagSampler {
    pub fn init(g: &PartiallyDirectedGraph) -> Self {
        let undirected_subgraph = g.undirected_subgraph();
        let directed_subgraph = g.directed_subgraph();
        let (subgraphs, comps) = undirected_subgraph.connected_components();
        let mut vertex_comp = vec![0; g.n];
        for (i, comp) in comps.iter().enumerate() {
            for &v in comp.iter() {
                vertex_comp[v] = i;
            }
        }

        // TODO: handle single vertex comps differently?
        let mut visited = vec![false; comps.len()];
        let mut ordered_comps = Vec::new();
        let mut samplers = Vec::new();
        for v in directed_subgraph.topological_order() {
            let comp_id = vertex_comp[v];
            if !visited[comp_id] {
                ordered_comps.push(comps[comp_id].clone()); // without clone?
                samplers.push(Sampler::init(&subgraphs[comp_id]));
                visited[comp_id] = true;
            }
        }
        CpdagSampler {
            samplers,
            ordered_comps,
            undirected_subgraph,
            directed_subgraph,
        }
    }

    // maybe use generics instead of ThreadRNG -> allow SeededRNGs as well
    pub fn sample_order(&self, rng: &mut ThreadRng) -> Vec<usize> {
        let mut order = Vec::new();
        for (i, sampler) in self.samplers.iter().enumerate() {
            sampler
                .sample_ordering(rng)
                .iter()
                .map(|&u| self.ordered_comps[i][u])
                .for_each(|u| order.push(u));
        }
        order
    }

    pub fn sample_dag(&self, rng: &mut ThreadRng) -> DirectedGraph {
        let order = self.sample_order(rng);
        let mut inv_order = vec![0; self.undirected_subgraph.n];
        order
            .iter()
            .enumerate()
            .for_each(|(i, &x)| inv_order[x] = i);
        let mut adjacency_list = self.directed_subgraph.to_adjacency_list();
        for u in 0..self.undirected_subgraph.n {
            for &v in self.undirected_subgraph.neighbors(u) {
                if inv_order[u] < inv_order[v] {
                    adjacency_list[u].push(v);
                }
            }
        }
        DirectedGraph::from_adjacency_list(adjacency_list)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use crate::{partially_directed_graph::PartiallyDirectedGraph, sample::CpdagSampler};

    #[test]
    fn sample_chordal_cpdag_basic_check() {
        let g = get_paper_graph();
        let sample_size = 10_000;
        let mut rng = rand::thread_rng();
        let sampler = CpdagSampler::init(&g);
        let amos: Vec<_> = (0..sample_size)
            .map(|_| sampler.sample_dag(&mut rng))
            .collect();
        assert_eq!(amos.len(), sample_size);
        let mut count_dags = HashMap::new();
        for a in amos.iter() {
            count_dags.entry(a).and_modify(|cnt| *cnt += 1).or_insert(1);
        }
        assert_eq!(count_dags.len(), 54);
        for (_, &cnt) in count_dags.iter() {
            assert!((cnt as i32).abs_diff(sample_size as i32 / 54) <= 50); // 54 is MEC size
        }
        let mut dags = HashSet::new();
        for a in amos.iter() {
            dags.insert(a.clone());
        }
        assert_eq!(dags.len(), 54);
    }

    #[test]
    fn sample_cpdag_basic_check() {
        let g = get_issue4_graph();
        let sample_size = 10_000;
        let mut rng = rand::thread_rng();
        let sampler = CpdagSampler::init(&g);
        let dags: Vec<_> = (0..sample_size)
            .map(|_| sampler.sample_dag(&mut rng))
            .collect();
        assert_eq!(dags.len(), sample_size);
        let mut count_dags = HashMap::new();
        for a in dags.iter() {
            count_dags.entry(a).and_modify(|cnt| *cnt += 1).or_insert(1);
        }
        assert_eq!(count_dags.len(), 10);
        let g = get_issue5_graph();
        let sample_size = 10_000;
        let mut rng = rand::thread_rng();
        let sampler = CpdagSampler::init(&g);
        let dags: Vec<_> = (0..sample_size)
            .map(|_| sampler.sample_dag(&mut rng))
            .collect();
        assert_eq!(dags.len(), sample_size);
        let mut count_dags = HashMap::new();
        for a in dags.iter() {
            count_dags.entry(a).and_modify(|cnt| *cnt += 1).or_insert(1);
        }
        assert_eq!(count_dags.len(), 44);
    }

    // TODO: test orders as well

    fn get_paper_graph() -> PartiallyDirectedGraph {
        PartiallyDirectedGraph::from_edge_list(
            vec![
                (0, 1),
                (1, 0),
                (0, 2),
                (2, 0),
                (1, 2),
                (2, 1),
                (1, 3),
                (3, 1),
                (1, 4),
                (4, 1),
                (1, 5),
                (5, 1),
                (2, 3),
                (3, 2),
                (2, 4),
                (4, 2),
                (2, 5),
                (5, 2),
                (3, 4),
                (4, 3),
                (4, 5),
                (5, 4),
            ],
            6,
        )
    }

    fn get_issue4_graph() -> PartiallyDirectedGraph {
        PartiallyDirectedGraph::from_edge_list(
            vec![
                (0, 1),
                (1, 0),
                (0, 2),
                (2, 0),
                (1, 2),
                (2, 1),
                (1, 3),
                (3, 1),
                (1, 4),
                (4, 1),
            ],
            5,
        )
    }

    fn get_issue5_graph() -> PartiallyDirectedGraph {
        PartiallyDirectedGraph::from_edge_list(
            vec![
                (9, 10),
                (9, 13),
                (9, 7),
                (10, 9),
                (10, 11),
                (10, 12),
                (13, 9),
                (4, 5),
                (4, 12),
                (5, 4),
                (0, 1),
                (0, 3),
                (1, 0),
                (1, 19),
                (6, 7),
                (6, 14),
                (6, 19),
                (7, 6),
                (7, 9),
                (7, 8),
                (14, 6),
                (14, 15),
                (8, 7),
                (8, 19),
                (16, 15),
                (16, 18),
                (16, 17),
                (15, 16),
                (15, 14),
                (18, 16),
                (18, 19),
                (11, 10),
                (11, 19),
                (3, 17),
                (3, 19),
                (2, 3),
            ],
            20,
        )
    }
}
