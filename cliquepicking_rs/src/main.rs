use std::time::Instant;

use cliquepicking::graph::Graph;
use cliquepicking::{count, sample};

fn main() {
    let g = Graph::from_stdin();
    println!("count {}", count::count_amos(&g));
    let now = Instant::now();
    println!("sample {:?}", sample::sample_amos(&g, 10)[0]);
    println!("total sample elapsed {}", now.elapsed().as_secs_f64());
}
