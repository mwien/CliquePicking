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
pub struct AliasTable {
    n: usize,
    prob: Vec<BigUint>, // maybe call this weight
    alias: Vec<usize>,
    total: BigUint,
}

#[derive(Debug)]
pub struct SamplingInfo {
    clique_tree: CliqueTree,
    separators: Vec<IndexSet>, // integrate separators and flowers into clique_tree
    flowers: Vec<IndexSet>,
    alias_tables: Vec<AliasTable>,
    forbidden_sets: Vec<Vec<(usize, usize, usize)>>,
}

fn alias_initialization(weights: &[BigUint], total: BigUint) -> AliasTable {
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
        weights_cloned[g] = weights_cloned[g].clone() + weights_cloned[l].clone() - total.clone();
        if weights_cloned[g] < total {
            small.push(g);
        } else {
            large.push(g);
        }
    }

    for &g in &large {
        prob[g] = total.clone();
    }

    AliasTable {
        n,
        prob,
        alias,
        total,
    }
}

fn draw_alias(alias_table: &AliasTable, rng: &mut ThreadRng) -> usize {
    let i = rng.gen_range(0..alias_table.n);
    let x = rng.gen_biguint_below(&alias_table.total);
    if x < alias_table.prob[i] {
        i
    } else {
        alias_table.alias[i]
    }
}

// duplicate code from count.rs
pub fn count_traversal(
    i: usize,
    visited: &mut LazyTokens,
    considered: &mut LazyTokens,
    pre: &mut Vec<BigUint>,
    flower: &IndexSet,
    memoization: &mut Memoization,
    alias_tables: &mut Vec<AliasTable>,
    clique_tree: &CliqueTree,
    separators: &Vec<IndexSet>,
    flowers: &Vec<IndexSet>,
    forbidden_sets: &Vec<Vec<(usize, usize, usize)>>,
) -> BigUint {
    visited.set(i);
    let mut product = BigUint::one();
    for &j in clique_tree.tree.neighbors(i) {
        if !flower.contains(j) {
            continue;
        }
        let edge_id = clique_tree.get_edge_id(i, j);
        if !visited.check(j) && !considered.check(j) {
            if pre[edge_id] != BigUint::ZERO {
                visited.set(j);
                product *= pre[edge_id].clone();
            } else {
                let next_flower_result = count_sampling_info(
                    edge_id,
                    visited,
                    considered,
                    memoization,
                    alias_tables,
                    clique_tree,
                    separators,
                    flowers,
                    forbidden_sets,
                );
                for &new_clique_id in &flowers[edge_id] {
                    considered.set(new_clique_id);
                }
                let remaining_subtree_result = count_traversal(
                    j,
                    visited,
                    considered,
                    pre,
                    flower,
                    memoization,
                    alias_tables,
                    clique_tree,
                    separators,
                    flowers,
                    forbidden_sets,
                );
                pre[edge_id] = next_flower_result * remaining_subtree_result;
                product *= pre[edge_id].clone();
            }
        } else if !visited.check(j) {
            product *= count_traversal(
                j,
                visited,
                considered,
                pre,
                flower,
                memoization,
                alias_tables,
                clique_tree,
                separators,
                flowers,
                forbidden_sets,
            );
        }
    }
    product
}

fn count_sampling_info(
    subproblem: usize,
    visited: &mut LazyTokens,
    considered: &mut LazyTokens,
    memoization: &mut Memoization,
    alias_tables: &mut Vec<AliasTable>,
    clique_tree: &CliqueTree,
    separators: &Vec<IndexSet>,
    flowers: &Vec<IndexSet>,
    forbidden_sets: &Vec<Vec<(usize, usize, usize)>>,
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
            * count_traversal(
                clique_id,
                visited,
                considered,
                &mut pre,
                flower,
                memoization,
                alias_tables,
                clique_tree,
                separators,
                flowers,
                forbidden_sets,
            );

        visited.restore();
        considered.restore();

        sum += product.clone();
        amos_per_clique[i] = product;
    }
    memoization.count[subproblem].clone_from(&sum);
    alias_tables[subproblem] = alias_initialization(&amos_per_clique, sum.clone()); // could also
                                                                                    // pass reference of sum
    sum
}

fn count_amos_sampling_info(g: &Graph) -> SamplingInfo {
    // compute the clique tree of G
    let clique_tree = CliqueTree::from(g); // TODO: time this
                                           //
                                           // compute all separators (one for each edge) and map from/to ordering
                                           // store as index set
    let mut separators = clique_tree.separators();
    separators.push(IndexSet::from_sorted(Vec::new())); // bake this into separator and flowers method

    // compute flowers of G
    let mut flowers = clique_tree.flowers(&separators);
    flowers.push(IndexSet::from_sorted((0..clique_tree.tree.n).collect()));

    let forbidden_sets = clique_tree.forbidden_sets(&separators, &flowers);

    let mut memoization = Memoization::new(clique_tree.tree.n, g.n);

    let mut visited = LazyTokens::new(clique_tree.tree.n);
    let mut considered = LazyTokens::new(clique_tree.tree.n);

    let mut alias_tables = vec![
        AliasTable {
            n: 0,
            prob: Vec::new(),
            alias: Vec::new(),
            total: BigUint::ZERO
        };
        flowers.len()
    ]; // could improve this

    // call recursive counting function
    count_sampling_info(
        flowers.len() - 1,
        &mut visited,
        &mut considered,
        &mut memoization,
        &mut alias_tables,
        &clique_tree,
        &separators,
        &flowers,
        &forbidden_sets,
    );

    SamplingInfo {
        clique_tree,
        separators,
        flowers,
        alias_tables,
        forbidden_sets,
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

fn draw_allowed_permutation(
    clique: &IndexSet,
    helper: &mut [usize],
    forbidden_prefixes: &Vec<IndexSet>,
) -> Vec<usize> {
    // first extract smallest set a vertex in clique appears in
    for &u in clique {
        helper[u] = clique.len();
    }
    // overwriting should be correct behaviour as forbidden prefixes are ordered from biggest to
    // smallest
    for forbidden_prefix in forbidden_prefixes {
        for &u in forbidden_prefix {
            helper[u] = forbidden_prefix.len() - 1;
        }
    }

    let mut rng = rand::thread_rng();

    loop {
        let mut perm = clique.clone().to_vec();
        perm.shuffle(&mut rng);
        if is_allowed(&perm, helper) {
            return perm;
        }
    }
}

fn sample_ordering_from_info(
    subproblem: usize,
    pos: &mut Vec<usize>,
    visited: &mut LazyTokens,
    considered: &mut LazyTokens,
    sampling_info: &SamplingInfo,
    rng: &mut ThreadRng,
) -> Vec<usize> {
    let mut ordering = Vec::new();

    let clique_id = sampling_info.flowers[subproblem]
        .get(draw_alias(&sampling_info.alias_tables[subproblem], rng)); // maybe one-liner
    let clique = &sampling_info.clique_tree.cliques[clique_id];
    let flower = &sampling_info.flowers[subproblem];
    let separator = &sampling_info.separators[subproblem];

    let remaining_clique_vertices = clique.set_difference(separator);
    let mut forbidden_prefixes = Vec::new();
    for &(u, v, size) in &sampling_info.forbidden_sets[clique_id] {
        if !flower.contains(u) || !flower.contains(v) {
            break;
        }
        if size > separator.len() {
            forbidden_prefixes.push(
                sampling_info.separators[sampling_info.clique_tree.get_edge_id(u, v)]
                    .set_difference(separator),
            );
        } else {
            break;
        }
    }
    let mut clique_ordering =
        draw_allowed_permutation(&remaining_clique_vertices, pos, &forbidden_prefixes);
    ordering.append(&mut clique_ordering);

    // could also pre-compute subproblems however that would increase
    // the run-time of counting for sparse graphs to n^3
    // precomputation would likely only improve run-time for finding ordering from O(n+m) to O(n)
    let flower = &sampling_info.flowers[subproblem];
    let mut queue = VecDeque::new();
    queue.push_back(clique_id);
    visited.prepare();
    considered.prepare();
    visited.set(clique_id);
    considered.set(clique_id);
    while !queue.is_empty() {
        let u = queue.pop_front().unwrap();
        for &v in sampling_info.clique_tree.tree.neighbors(u) {
            if !flower.contains(v) {
                continue;
            }
            if !visited.check(v) {
                queue.push_back(v);
                visited.set(v);
            }
            if !considered.check(v) {
                let new_flower_id = sampling_info.clique_tree.get_edge_id(u, v);
                ordering.append(&mut sample_ordering_from_info(
                    new_flower_id,
                    pos,
                    visited,
                    considered,
                    sampling_info,
                    rng,
                ));
                for &new_clique_id in &sampling_info.flowers[new_flower_id] {
                    considered.set(new_clique_id);
                }
            }
        }
    }
    visited.restore();
    considered.restore();
    ordering
}

// TODO: use sample_amo_orders
pub fn sample_amos(g: &Graph, k: usize) -> Vec<DirectedGraph> {
    let sampling_info = count_amos_sampling_info(g);
    let mut samples = Vec::new();
    let mut rng = rand::thread_rng();
    for _ in 0..k {
        let mut visited = LazyTokens::new(sampling_info.clique_tree.tree.n);
        let mut considered = LazyTokens::new(sampling_info.clique_tree.tree.n);
        let mut pos = vec![0; g.n];
        let ordering = sample_ordering_from_info(
            sampling_info.flowers.len() - 1,
            &mut pos,
            &mut visited,
            &mut considered,
            &sampling_info,
            &mut rng,
        );
        samples.push(g.orient(&ordering));
    }
    samples
}

pub fn sample_chordal(g: &Graph, k: usize) -> Vec<DirectedGraph> {
    let (components, vertices) = g.connected_components();
    let mut adjacency_lists = vec![vec![Vec::new(); g.n]; k];
    for (i, component) in components.iter().enumerate() {
        let component_samples = sample_amos(component, k);
        for (j, sample) in component_samples.iter().enumerate() {
            for u in 0..sample.n {
                for &v in sample.outneighbors(u) {
                    adjacency_lists[j][vertices[i][u]].push(vertices[i][v]);
                }
            }
        }
    }
    adjacency_lists
        .into_iter()
        .map(DirectedGraph::from_adjacency_list)
        .collect()
}

// there are unnecessary allocations/conversions here, maybe optimize this at some point
pub fn sample_cpdag(g: &PartiallyDirectedGraph, k: usize) -> Vec<DirectedGraph> {
    let undirected_subgraph = g.undirected_subgraph();
    let directed_subgraph = g.directed_subgraph();
    let samples = sample_chordal(&undirected_subgraph, k);
    let mut adjacency_lists: Vec<Vec<Vec<usize>>> =
        samples.into_iter().map(|x| x.to_adjacency_list()).collect();
    for u in 0..g.n {
        for &v in directed_subgraph.outneighbors(u) {
            adjacency_lists.iter_mut().for_each(|l| l[u].push(v));
        }
    }
    adjacency_lists
        .into_iter()
        .map(DirectedGraph::from_adjacency_list)
        .collect()
}

fn sample_amo_orders(g: &Graph, k: usize) -> Vec<Vec<usize>> {
    let sampling_info = count_amos_sampling_info(g);
    let mut orders = Vec::new();
    let mut rng = rand::thread_rng(); // for small graphs, seeding the RNG is the costliest operation
    for _ in 0..k {
        let mut visited = LazyTokens::new(sampling_info.clique_tree.tree.n);
        let mut considered = LazyTokens::new(sampling_info.clique_tree.tree.n);
        let mut pos = vec![0; g.n];
        let ordering = sample_ordering_from_info(
            sampling_info.flowers.len() - 1,
            &mut pos,
            &mut visited,
            &mut considered,
            &sampling_info,
            &mut rng,
        );
        orders.push(ordering);
    }
    orders
}

fn sample_chordal_orders(g: &Graph, k: usize) -> Vec<Vec<Vec<usize>>> {
    let (components, vertices) = g.connected_components();
    let mut component_orders = vec![Vec::new(); k];
    for (i, component) in components.iter().enumerate() {
        for (j, component_order) in sample_amo_orders(component, k).iter().enumerate() {
            component_orders[j].push(component_order.iter().map(|&x| vertices[i][x]).collect());
        }
    }
    component_orders
}

fn construct_order(g: &DirectedGraph, component_orders: &[Vec<usize>]) -> Vec<usize> {
    let mut comp = vec![0; g.n];
    let mut vis_comp = vec![false; component_orders.len()];
    // extract in which compoment which vertex is
    for (i, component_order) in component_orders.iter().enumerate() {
        for &v in component_order.iter() {
            comp[v] = i;
        }
    }

    let mut order = Vec::new();
    for &u in g.topological_order().iter() {
        if !vis_comp[comp[u]] {
            component_orders[comp[u]]
                .iter()
                .for_each(|&x| order.push(x));
            vis_comp[comp[u]] = true;
        }
    }
    order
}

pub fn sample_cpdag_orders(g: &PartiallyDirectedGraph, k: usize) -> Vec<Vec<usize>> {
    let undirected_subgraph = g.undirected_subgraph();
    let directed_subgraph = g.directed_subgraph();
    let component_orders = sample_chordal_orders(&undirected_subgraph, k);
    component_orders
        .iter()
        .map(|c| construct_order(&directed_subgraph, c))
        .collect()
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use crate::graph::Graph;

    #[test]
    fn sample_amos_basic_check() {
        let g = Graph::from_edge_list(
            vec![
                (0, 1),
                (0, 2),
                (1, 2),
                (1, 3),
                (1, 4),
                (1, 5),
                (2, 3),
                (2, 4),
                (2, 5),
                (3, 4),
                (4, 5),
            ],
            6,
        );
        let sample_size = 10_000;
        let amos = super::sample_amos(&g, sample_size);
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
}
