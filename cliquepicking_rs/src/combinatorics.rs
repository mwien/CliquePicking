use num_bigint::BigUint;
use num_traits::One;

use crate::memoization::Memoization;

pub fn factorial(n: usize, memoization: &mut [BigUint]) -> BigUint {
    if memoization[n] != BigUint::ZERO {
        return memoization[n].clone();
    }
    let mut result = BigUint::one();
    for i in 1..n + 1 {
        result *= i;
    }
    memoization[n].clone_from(&result);
    result
}

// TODO: maybe implement without recursion?
pub fn rho(x: &[usize], memoization: &mut Memoization) -> BigUint {
    let x_vec = x.to_vec();
    if let Some(res) = memoization.rho.get(&x_vec) {
        return res.clone();
    }
    // what if x is empty?
    let mut result = factorial(x[0], &mut memoization.factorial);
    for i in 1..x.len() {
        result -= factorial(x[0] - x[i], &mut memoization.factorial) * rho(&x[i..], memoization);
    }
    memoization.rho.insert(x_vec, result.clone());
    result
}
