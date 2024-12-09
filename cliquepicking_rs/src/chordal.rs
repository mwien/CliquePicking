use crate::graph::Graph;

pub fn mcs(g: &Graph) -> Vec<usize> {
    let mut ordering = Vec::new();
    let mut sets: Vec<Vec<usize>> = vec![Vec::new(); g.n];
    let mut cardinality = vec![0; g.n];
    let mut max_cardinality = 0;

    sets[0] = (0..g.n).collect();

    let mut idx = 0;
    while idx < g.n {
        while max_cardinality > 0 && sets[max_cardinality].is_empty() {
            max_cardinality -= 1;
        }
        let u = sets[max_cardinality].pop().unwrap();
        if cardinality[u] == usize::MAX {
            continue;
        }
        idx += 1;
        ordering.push(u);
        cardinality[u] = usize::MAX;
        for &v in g.neighbors(u) {
            if cardinality[v] < g.n {
                cardinality[v] += 1;
                sets[cardinality[v]].push(v);
            }
        }
        max_cardinality += 1;
    }
    ordering
}

#[cfg(test)]
mod tests {
    use super::mcs;
    use crate::graph::Graph;
    use crate::utils;

    // Could put this in module above if used for chordality test. Current standpoint is that
    // chordality is assumed for the appropriate functions in this crate but not actively tested.

    // Roughly follows "Test for zero fill-in" (Simple linear-time algorithms to test chordality of
    // graphs, test acyclicity of hypergraphs and selectively reduce acyclic hypergraphs; Tarjan,
    // Yannakakis; p. 571).
    fn is_peo(g: &Graph, ordering: &Vec<usize>) -> bool {
        let inv_ordering = utils::inverse_permutation(ordering);
        // find first neighbor before each vertex in ordering
        let mut prev_neighbor = vec![0; g.n];
        for (i, &u) in ordering.iter().enumerate() {
            for &v in g.neighbors(u) {
                if inv_ordering[v] > i {
                    prev_neighbor[v] = u;
                }
            }
        }
        let mut token = vec![usize::MAX; g.n];
        for (i, &u) in ordering.iter().enumerate() {
            token[u] = u;
            // put token on each neighbor coming after u in ordering
            for &v in g.neighbors(u) {
                if inv_ordering[v] > i {
                    token[v] = u;
                }
            }
            // check for peo violation
            for &v in g.neighbors(u) {
                if inv_ordering[v] > i && token[prev_neighbor[v]] != u {
                    return false;
                }
            }
        }
        true
    }

    #[test]
    fn is_peo_basic_check() {
        let g =
            Graph::from_adjacency_list(vec![vec![1, 2, 3], vec![0, 3], vec![0, 3], vec![0, 1, 2]]);
        let peo = vec![0, 1, 3, 2];
        let not_peo = vec![0, 1, 2, 3];
        assert!(is_peo(&g, &peo));
        assert!(!is_peo(&g, &not_peo));
    }

    #[test]
    fn mcs_basic_check() {
        let g =
            Graph::from_adjacency_list(vec![vec![1, 2, 3], vec![0, 3], vec![0, 3], vec![0, 1, 2]]);
        let ordering = mcs(&g);
        assert!(is_peo(&g, &ordering));
    }
}
