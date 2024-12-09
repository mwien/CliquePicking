use rug::Integer;
use std::collections::HashMap;

#[derive(Debug)]
pub struct Memoization {
    pub count: Vec<Integer>,
    pub rho: HashMap<Vec<usize>, Integer>,
    pub factorial: Vec<Integer>,
}

impl Memoization {
    pub fn new(num_cliques: usize, n: usize) -> Memoization {
        Memoization {
            count: vec![Integer::from(0); 2 * num_cliques - 1],
            rho: HashMap::new(),
            factorial: vec![Integer::from(0); n + 1],
        }
    }
}
