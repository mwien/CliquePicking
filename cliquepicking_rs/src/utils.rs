use std::io::{self, BufRead};

pub fn inverse_permutation(permutation: &[usize]) -> Vec<usize> {
    let mut inverse_permutation = vec![0; permutation.len()];
    for (i, &el) in permutation.iter().enumerate() {
        inverse_permutation[el] = i;
    }
    inverse_permutation
}

pub fn edge_list_from_stdin() -> (Vec<(usize, usize)>, usize) {
    let mut iterator = io::stdin().lock().lines();
    let first_line = iterator.next().unwrap().unwrap();
    let (first, _) = first_line.split_once(' ').unwrap();
    let n = first.parse::<usize>().unwrap();
    let mut edge_list = Vec::new();
    for line in iterator {
        let next_line = line.unwrap();
        let (first, second) = next_line.split_once(' ').unwrap();
        edge_list.push((
            first.parse::<usize>().unwrap() - 1,
            second.parse::<usize>().unwrap() - 1,
        ));
    }
    (edge_list, n)
}
