use std::usize;

use crate::{
    directed_graph::DirectedGraph, graph::Graph, partially_directed_graph::PartiallyDirectedGraph,
};

struct McsState {
    ordering: Vec<usize>,
    sets: Vec<Vec<usize>>,
    cardinality: Vec<usize>,
    max_cardinality: usize,
    position: usize,
}

impl McsState {
    pub fn new(n: usize) -> McsState {
        let mut sets = vec![Vec::new(); n];
        sets[0] = (0..n).collect();
        McsState {
            ordering: Vec::new(),
            sets,
            cardinality: vec![0; n],
            max_cardinality: 0,
            position: 0,
        }
    }
}

fn visit(g: &Graph, state: &mut McsState, u: usize) {
    state.position += 1;
    state.ordering.push(u);
    state.cardinality[u] = usize::MAX;
    for &v in g.neighbors(u) {
        if state.cardinality[v] < g.n {
            state.cardinality[v] += 1;
            state.sets[state.cardinality[v]].push(v);
        }
    }
    state.max_cardinality += 1;
    while state.max_cardinality > 0 && state.sets[state.max_cardinality].is_empty() {
        state.max_cardinality -= 1;
    }
}

fn unvisit(g: &Graph, state: &mut McsState, u: usize) {
    state.cardinality[u] = usize::MAX; // TODO:
    state.sets[state.cardinality[u]].push(u);
    state.position -= 1;

    for &v in g.neighbors(u).rev() {
        if state.cardinality[v] < g.n {
            // should always be true
            assert_eq!(state.sets[state.cardinality[v]].pop().unwrap(), v);
            state.cardinality[v] -= 1;
            state.sets[state.cardinality[v]].push(v);
        }
    }

    state.max_cardinality = state.cardinality[u];
}

fn reach(g: &Graph, s: usize) -> Vec<usize> {
    let mut visited = vec![false; g.n];
    visited[s] = true;
    let mut queue = vec![s];

    while !queue.is_empty() {
        let u = queue.pop().unwrap();
        for &v in g.neighbors(u) {
            if !visited[v] {
                queue.push(v);
                visited[v] = true;
            }
        }
    }

    visited
        .iter()
        .enumerate()
        .filter(|(_, &val)| val)
        .map(|(i, _)| i)
        .collect()
}

fn rec_list_chordal_orders(g: &Graph, orders: &mut Vec<Vec<usize>>, state: &mut McsState) {
    if state.position == g.n {
        orders.push(state.ordering.clone());
    }

    // do this better
    let u = loop {
        let next_vertex = state.sets[state.max_cardinality].pop().unwrap();
        // use Result instead of this hack
        if state.cardinality[next_vertex] == usize::MAX {
            break next_vertex;
        }
    };

    visit(g, state, u);
    rec_list_chordal_orders(g, orders, state);
    unvisit(g, state, u);

    let reachable = reach(g, u);

    for x in reachable {
        if x == u {
            continue;
        }
        visit(g, state, u);
        rec_list_chordal_orders(g, orders, state);
        unvisit(g, state, u);
    }
}

fn list_chordal_orders(g: &Graph) -> Vec<Vec<usize>> {
    let mut orders = Vec::new();
    rec_list_chordal_orders(g, &mut orders, &mut McsState::new(g.n));
    orders
}

fn sort_order(d: &DirectedGraph, cmp: &Vec<usize>, order: &Vec<usize>) -> Vec<usize> {
    let mut component_no = vec![usize::MAX; *cmp.iter().max().unwrap() + 1];
    let mut sorted_order = Vec::new();

    // TODO: check this

    let to = d.topological_order();
    let mut found_comps = 0;
    for &u in to.iter() {
        if component_no[cmp[u]] == usize::MAX {
            component_no[cmp[u]] = found_comps;
            found_comps += 1;
            sorted_order.push(vec![u]);
        }
    }

    for &u in order.iter() {
        let cmp_u = component_no[cmp[u]];
        sorted_order[cmp_u].push(u);
    }

    sorted_order.into_iter().flatten().collect()
}

pub fn list_cpdag_orders(g: &PartiallyDirectedGraph) -> Vec<Vec<usize>> {
    let undirected_subgraph = g.undirected_subgraph();
    let directed_subgraph = g.directed_subgraph();
    let unsorted_orders = list_chordal_orders(&undirected_subgraph);

    // could use a method which only returns list of vertex lists
    let (_, vertices) = undirected_subgraph.connected_components();
    let mut cmp = vec![0; g.n];
    vertices
        .iter()
        .enumerate()
        .for_each(|(i, l)| l.into_iter().for_each(|&v| cmp[v] = i));

    unsorted_orders
        .iter()
        .map(|order| sort_order(&directed_subgraph, &cmp, order))
        .collect()
}
