use num_bigint::BigUint;
use num_traits::One;

use crate::clique_tree::CliqueTree;
use crate::combinatorics;
use crate::graph::Graph;
use crate::index_set::IndexSet;
use crate::lazy_tokens::LazyTokens;
use crate::memoization::Memoization;
use crate::partially_directed_graph::PartiallyDirectedGraph;

pub fn count_traversal(
    i: usize,
    visited: &mut LazyTokens,
    considered: &mut LazyTokens,
    pre: &mut Vec<BigUint>,
    flower: &IndexSet,
    memoization: &mut Memoization,
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
                let next_flower_result = count(
                    edge_id,
                    visited,
                    considered,
                    memoization,
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
                clique_tree,
                separators,
                flowers,
                forbidden_sets,
            );
        }
    }
    product
}

fn count(
    subproblem: usize,
    visited: &mut LazyTokens,
    considered: &mut LazyTokens,
    memoization: &mut Memoization,
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
    if flower.len() == 1 {
        // flower consists of single clique
        let res = combinatorics::factorial(
            clique_tree.cliques[flower.first().unwrap()].len() - separator.len(),
            &mut memoization.factorial,
        );
        memoization.count[subproblem].clone_from(&res);
        return res;
    }

    let mut sum = BigUint::ZERO;
    let mut pre = vec![BigUint::ZERO; 2 * (clique_tree.tree.n - 1)];
    for &clique_id in flower {
        let mut forbidden_sizes = Vec::new();
        forbidden_sizes.push(clique_tree.cliques[clique_id].len() - separator.len());
        for &(u, v, size) in &forbidden_sets[clique_id] {
            if !flower.contains(u) || !flower.contains(v) {
                break;
            }
            assert!(size > separator.len()); // TODO: maybe equality can occur
            forbidden_sizes.push(size - separator.len());
        }
        let phi = combinatorics::rho(&forbidden_sizes, memoization);

        visited.prepare();
        considered.prepare();
        visited.set(clique_id);
        considered.set(clique_id);

        let product = count_traversal(
            clique_id,
            visited,
            considered,
            &mut pre,
            flower,
            memoization,
            clique_tree,
            separators,
            flowers,
            forbidden_sets,
        );

        visited.restore();
        considered.restore();

        sum += phi * product;
    }
    memoization.count[subproblem].clone_from(&sum);
    sum
}

pub fn count_amos(g: &Graph) -> BigUint {
    //if g.m == g.n-1 { return Integer::from(g.n); }
    //if g.m == g.n { return Integer::from(2*g.n); }
    //let num_possible_edges = g.n * (g.n - 1) / 2;
    //if g.m == num_possible_edges - 2 { return (g.n * (g.n-1) - 4) * combinatorics::factorial(g.n-3, &mut Memoization::new(0, g.n)); }
    //if g.m == num_possible_edges - 1 { return (2 * g.n - 3) * combinatorics::factorial(g.n-2, &mut Memoization::new(0, g.n)); }
    //if g.m == num_possible_edges { return combinatorics::factorial(g.n, &mut Memoization::new(0, g.n)); }

    // compute the clique tree of G
    let clique_tree = CliqueTree::from(g);

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

    // call recursive counting function
    count(
        flowers.len() - 1,
        &mut visited,
        &mut considered,
        &mut memoization,
        &clique_tree,
        &separators,
        &flowers,
        &forbidden_sets,
    )
}

// call count_amos for each connected component
pub fn count_chordal(g: &Graph) -> BigUint {
    let (components, _) = g.connected_components();
    components.iter().map(count_amos).product()
}

// remove directed edges and call count_chordal
pub fn count_cpdag(g: &PartiallyDirectedGraph) -> BigUint {
    let undirected_subgraph = g.undirected_subgraph();
    count_chordal(&undirected_subgraph)
}

// Currently not implemented and not actively planned:
// pub fn count_mpdag(g: &Graph) -> Integer {
//  // implement adaption for background knowledge
// }
//
// Note: It is NOT (!) checked whether g satisfies the assumed properties. If in count_chordal the
// given graph is not chordal or in count_cpdag the graph is not a CPDAG (i.e., not satisfying the properties
// given by Andersson et al. in 'A characterization of Markov equivalence classes for acyclic
// digraphs' (1997)), then the result may be WRONG or the program might CRASH.

#[cfg(test)]
mod tests {
    use num_bigint::ToBigUint;

    use crate::{graph::Graph, partially_directed_graph::PartiallyDirectedGraph};

    #[test]
    fn count_amos_basic_check() {
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
        assert_eq!(super::count_amos(&g), 54.to_biguint().unwrap());
    }

    #[test]
    fn count_chordal_basic_check() {
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
                (7, 8),
            ],
            9,
        );
        assert_eq!(super::count_chordal(&g), 108.to_biguint().unwrap());
    }

    #[test]
    fn count_cpdag_basic_check() {
        let g = PartiallyDirectedGraph::from_edge_list(
            vec![
                (0, 2),
                (1, 2),
                (2, 3),
                (2, 4),
                (2, 5),
                (3, 4),
                (4, 3),
                (4, 5),
                (4, 6),
                (4, 7),
                (5, 4),
                (6, 7),
                (7, 6),
            ],
            8,
        );
        assert_eq!(super::count_cpdag(&g), 6.to_biguint().unwrap());
    }
}
