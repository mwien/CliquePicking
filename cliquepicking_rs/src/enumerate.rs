use crate::{
    directed_graph::DirectedGraph, graph::Graph, partially_directed_graph::PartiallyDirectedGraph,
};

#[derive(Debug)]
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
    state.cardinality[u] = usize::MAX; // TODO: use Option to encode this
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

fn unvisit(g: &Graph, state: &mut McsState, u: usize, last_cardinality: usize) {
    state.position -= 1;
    state.ordering.pop();
    state.cardinality[u] = last_cardinality;
    state.sets[state.cardinality[u]].push(u);

    for &v in g.neighbors(u).rev() {
        // TODO: sets will get bigger and bigger -> cleanup?
        if state.cardinality[v] < g.n {
            state.cardinality[v] -= 1;
            state.sets[state.cardinality[v]].push(v);
        }
    }

    state.max_cardinality = state.cardinality[u];
}

fn reach(g: &Graph, st: &[usize], s: usize) -> Vec<usize> {
    let mut visited = vec![false; g.n];
    visited[s] = true;
    let mut blocked = vec![true; g.n];
    st.iter().for_each(|&v| blocked[v] = false);
    let mut queue = vec![s];

    while let Some(u) = queue.pop() {
        for &v in g.neighbors(u) {
            if !visited[v] && !blocked[v] {
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
        return;
    }

    // do this better
    let u = loop {
        while state.max_cardinality > 0 && state.sets[state.max_cardinality].is_empty() {
            state.max_cardinality -= 1;
        }
        let next_vertex = state.sets[state.max_cardinality].pop().unwrap();
        // use Result instead of this hack
        if state.cardinality[next_vertex] == state.max_cardinality {
            break next_vertex;
        }
    };

    let last_cardinality = state.cardinality[u];
    visit(g, state, u);
    rec_list_chordal_orders(g, orders, state);
    unvisit(g, state, u, last_cardinality);

    let st: Vec<_> = state.sets[state.max_cardinality]
        .iter()
        .copied()
        .filter(|&v| state.max_cardinality == state.cardinality[v])
        .collect();
    let reachable = reach(g, &st, u);

    for x in reachable {
        if x == u || state.cardinality[x] != state.max_cardinality {
            continue;
        }
        let last_cardinality = state.cardinality[x];
        visit(g, state, x);
        rec_list_chordal_orders(g, orders, state);
        unvisit(g, state, x, last_cardinality);
    }
}

fn list_chordal_orders(g: &Graph) -> Vec<Vec<usize>> {
    let mut orders = Vec::new();
    rec_list_chordal_orders(g, &mut orders, &mut McsState::new(g.n));
    orders
}

fn sort_order(d: &DirectedGraph, cmp: &[usize], order: &[usize]) -> Vec<usize> {
    let mut component_no = vec![usize::MAX; *cmp.iter().max().unwrap() + 1];
    let mut sorted_order = Vec::new();

    let to = d.topological_order();
    let mut found_comps = 0;
    for &u in to.iter() {
        if component_no[cmp[u]] == usize::MAX {
            component_no[cmp[u]] = found_comps;
            found_comps += 1;
            sorted_order.push(Vec::new());
        }
    }

    for &u in order.iter() {
        let cmp_u = component_no[cmp[u]];
        sorted_order[cmp_u].push(u);
    }

    sorted_order.into_iter().flatten().collect()
}

// TODO: rename
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
        .for_each(|(i, l)| l.iter().for_each(|&v| cmp[v] = i));

    unsorted_orders
        .iter()
        .map(|order| sort_order(&directed_subgraph, &cmp, order))
        .collect()
}

pub fn list_cpdag(g: &PartiallyDirectedGraph) -> Vec<DirectedGraph> {
    let undirected_subgraph = g.undirected_subgraph();
    let directed_subgraph = g.directed_subgraph();

    let mut dags = Vec::new();
    for order in list_cpdag_orders(g).iter() {
        let mut position = vec![0; order.len()];
        order.iter().enumerate().for_each(|(i, &v)| position[v] = i);
        let mut dag_edge_list = directed_subgraph.to_edge_list();
        for &(u, v) in undirected_subgraph.to_edge_list().iter() {
            if u > v {
                continue;
            }
            if position[u] < position[v] {
                dag_edge_list.push((u, v));
            } else {
                dag_edge_list.push((v, u));
            }
        }
        dags.push(DirectedGraph::from_edge_list(dag_edge_list, order.len()));
    }
    dags
}

#[cfg(test)]
mod tests {

    use crate::partially_directed_graph::PartiallyDirectedGraph;

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

    fn get_basic_graph() -> PartiallyDirectedGraph {
        PartiallyDirectedGraph::from_edge_list(
            vec![(0, 1), (1, 0), (1, 2), (2, 1), (0, 3), (2, 3)],
            4,
        )
    }

    #[test]
    fn list_cpdag_basic_check() {
        let dags = super::list_cpdag(&get_paper_graph());
        assert_eq!(dags.len(), 54);
        let dags = super::list_cpdag(&get_basic_graph());
        assert_eq!(dags.len(), 3);
        // TODO: better tests
    }

    #[test]
    fn list_cpdag_orders_basic_check() {
        let orders = super::list_cpdag_orders(&get_paper_graph());
        assert_eq!(orders.len(), 54);
        let orders = super::list_cpdag_orders(&get_basic_graph());
        assert_eq!(orders.len(), 3);
        // TODO: better tests
    }
}
