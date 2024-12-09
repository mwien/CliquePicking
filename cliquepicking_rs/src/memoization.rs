use num_bigint::BigUint;
use std::collections::HashMap;

#[derive(Debug)]
pub struct Memoization {
    pub count: Vec<BigUint>,
    pub rho: HashMap<Vec<usize>, BigUint>,
    pub factorial: Vec<BigUint>,
}

impl Memoization {
    pub fn new(num_cliques: usize, n: usize) -> Memoization {
        Memoization {
            count: vec![BigUint::ZERO; 2 * num_cliques - 1],
            rho: HashMap::new(),
            factorial: vec![BigUint::ZERO; n + 1],
        }
    }
}
