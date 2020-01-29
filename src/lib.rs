use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use std::collections::HashMap;

fn get_constraints(
    constraint_indices: impl Iterator<Item = (usize, usize)>,
    num_constraints: usize,
) -> Vec<Vec<usize>> {
    let mut constraints: Vec<Vec<usize>> =
        (0..num_constraints).map(|_| Vec::<usize>::new()).collect();

    for (index, constraint) in
        constraint_indices.filter(|(index, constraint)| constraint < &num_constraints)
    {
        constraints
            .get_mut(constraint)
            .expect(&format!["No constraint at index {}", constraint])
            .push(index)
    }

    constraints.into_iter().collect()
}

#[pyfunction]
fn rusty_121_t2s_node(
    t_nodes: Vec<usize>,
    node_matchings: Vec<(usize, Option<usize>)>,
) -> PyResult<Vec<Vec<usize>>> {
    let index_map: HashMap<usize, usize> = t_nodes
        .clone()
        .into_iter()
        .enumerate()
        .map(|(i, x)| (x, i))
        .collect();

    let indices: Vec<usize> = node_matchings
        .iter()
        .map(|(s_node, t_node)| match t_node {
            Some(t_node) => *index_map
                .get(&t_node)
                .expect(&format!["Index map has no entry for {}", t_node]),
            None => t_nodes.len(),
        })
        .collect();

    Ok(get_constraints(
        indices.into_iter().enumerate(),
        t_nodes.len(),
    ))
}

#[pyfunction]
fn rusty_12n_t2s_edge(
    t_edges: Vec<(usize, usize)>,
    edge_matchings: Vec<(usize, usize, usize, usize)>,
) -> PyResult<Vec<Vec<usize>>> {
    let index_map: HashMap<(usize, usize), usize> = t_edges
        .clone()
        .into_iter()
        .enumerate()
        .map(|(i, x)| (x, i))
        .collect();

    let indices: Vec<usize> = edge_matchings
        .iter()
        .map(|(s_u, s_v, t_u, t_v)| {
            *index_map
                .get(&(*t_u, *t_v))
                .expect(&format!["Index map has no entry for {:?}", (t_u, t_v)])
        })
        .collect();

    Ok(get_constraints(
        indices.into_iter().enumerate(),
        t_edges.len(),
    ))
}

#[pyfunction]
fn rusty_s_nodes(
    s_nodes: Vec<usize>,
    node_matchings: Vec<(usize, Option<usize>)>,
) -> PyResult<Vec<Vec<usize>>> {
    unimplemented![];
    Ok(vec![vec![1]])
}

/// This module is a python module implemented in Rust.
#[pymodule]
fn funlib_match_helpers(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(rusty_121_t2s_node))?;
    m.add_wrapped(wrap_pyfunction!(rusty_12n_t2s_edge))?;
    m.add_wrapped(wrap_pyfunction!(rusty_s_nodes))?;

    Ok(())
}
