#![feature(option_expect_none)]

#[macro_use]
use ndarray::ShapeError;
use ndarray::linalg::Dot;
use ndarray::{s, Array, ArrayD, ArrayViewD, Ix3, Ix4, IxDynImpl};
use numpy::{IntoPyArray, PyArrayDyn};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::{exceptions, PyErr, PyResult};

use petgraph::algo::min_spanning_tree;
use petgraph::data::*;
use petgraph::graph::NodeIndex;
use petgraph::visit::Dfs;
use petgraph::Graph;

use std;

use std::error;
use std::fmt;

use std::collections::hash_map::Entry::{Occupied, Vacant};
use std::collections::{BinaryHeap, HashMap, HashSet};

use std::hash::Hash;

mod scored;

use petgraph::algo::Measure;
use petgraph::visit::{EdgeRef, IntoEdges, VisitMap, Visitable};
use scored::MinScored;

#[derive(Debug)]
struct ShapeErrorWrapper {
    err: ShapeError,
}

impl fmt::Display for ShapeErrorWrapper {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.err.fmt(f)
    }
}

impl error::Error for ShapeErrorWrapper {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        self.err.source()
    }
}

impl std::convert::From<ShapeErrorWrapper> for PyErr {
    fn from(err: ShapeErrorWrapper) -> PyErr {
        exceptions::OSError::py_err(err.to_string())
    }
}

fn maximin_3d_tree(
    intensities: &Array<f64, Ix3>,
) -> Graph<(usize, usize, usize), (f64, f64), petgraph::Undirected> {
    let n_dim = intensities.ndim();
    let mut node_indices = Array::<usize, _>::from_elem(intensities.raw_dim(), usize::max_value());
    let mut complete: Graph<(usize, usize, usize), (f64, f64), petgraph::Undirected> =
        Graph::new_undirected();

    let mut node_index_map: HashMap<usize, (usize, usize, usize)> = HashMap::new();

    for (index, value) in intensities.indexed_iter() {
        let node_index = complete.add_node(index);
        node_index_map.insert(node_index.index(), index.clone());
        node_indices[index] = node_index.index();
        for i in 0..n_dim {
            let mut adj: Vec<usize> = (0..n_dim).map(|_| 0).collect();
            adj[i] = 1;
            let up_neighbor_index = (
                usize::wrapping_add(index.0, adj[0]),
                usize::wrapping_add(index.1, adj[1]),
                usize::wrapping_add(index.2, adj[2]),
            );
            let down_neighbor_index = (
                usize::wrapping_sub(index.0, adj[0]),
                usize::wrapping_sub(index.1, adj[1]),
                usize::wrapping_sub(index.2, adj[2]),
            );

            let up_index = node_indices.get(up_neighbor_index);
            let down_index = node_indices.get(down_neighbor_index);

            match up_index {
                Some(up_index) => {
                    if up_index < &usize::max_value() {
                        let min_intensity = f64::min(*value, intensities[up_neighbor_index]);
                        let max_intensity = f64::max(*value, intensities[up_neighbor_index]);
                        complete.add_edge(
                            node_index,
                            NodeIndex::new(*up_index),
                            (-min_intensity, -max_intensity),
                        );
                    }
                }
                None => (),
            }

            match down_index {
                Some(down_index) => {
                    if down_index < &usize::max_value() {
                        let min_intensity = f64::min(*value, intensities[down_neighbor_index]);
                        let max_intensity = f64::max(*value, intensities[down_neighbor_index]);
                        complete.add_edge(
                            node_index,
                            NodeIndex::new(*down_index),
                            (-min_intensity, -max_intensity),
                        );
                    }
                }
                None => (),
            }
        }
    }

    let mst: Graph<(usize, usize, usize), (f64, f64), petgraph::Undirected> =
        Graph::from_elements(min_spanning_tree(&complete));
    return mst;
}

fn masked_maximin_3d_tree(
    intensities: &Array<f64, Ix3>,
    mask: &Array<u8, Ix3>,
) -> (
    Graph<(usize, usize, usize), (f64, f64), petgraph::Undirected>,
    Vec<(usize, usize, usize)>,
) {
    let n_dim = intensities.ndim();
    let mut node_indices = Array::<usize, _>::from_elem(intensities.raw_dim(), usize::max_value());
    let mut complete: Graph<(usize, usize, usize), (f64, f64), petgraph::Undirected> =
        Graph::new_undirected();

    let mut points = Vec::new();

    for (index, value) in intensities.indexed_iter() {
        let node_index = complete.add_node(index);

        if *mask
            .get(index)
            .expect(&format!["Mask does not have a value at {:?}", index])
            > 0u8
        {
            points.push(index);
        }

        node_indices[index] = node_index.index();
        for i in 0..n_dim {
            let mut adj: Vec<usize> = (0..n_dim).map(|_| 0).collect();
            adj[i] = 1;
            let up_neighbor_index = (
                usize::wrapping_add(index.0, adj[0]),
                usize::wrapping_add(index.1, adj[1]),
                usize::wrapping_add(index.2, adj[2]),
            );
            let down_neighbor_index = (
                usize::wrapping_sub(index.0, adj[0]),
                usize::wrapping_sub(index.1, adj[1]),
                usize::wrapping_sub(index.2, adj[2]),
            );

            let up_index = node_indices.get(up_neighbor_index);
            let down_index = node_indices.get(down_neighbor_index);

            match up_index {
                Some(up_index) => {
                    if up_index < &usize::max_value() {
                        let min_intensity = f64::min(*value, intensities[up_neighbor_index]);
                        let max_intensity = f64::max(*value, intensities[up_neighbor_index]);
                        complete.add_edge(
                            node_index,
                            NodeIndex::new(*up_index),
                            (-min_intensity, -max_intensity),
                        );
                    }
                }
                None => (),
            }

            match down_index {
                Some(down_index) => {
                    if down_index < &usize::max_value() {
                        let min_intensity = f64::min(*value, intensities[down_neighbor_index]);
                        let max_intensity = f64::max(*value, intensities[down_neighbor_index]);
                        complete.add_edge(
                            node_index,
                            NodeIndex::new(*down_index),
                            (-min_intensity, -max_intensity),
                        );
                    }
                }
                None => (),
            }
        }
    }

    let mst: Graph<(usize, usize, usize), (f64, f64), petgraph::Undirected> =
        Graph::from_elements(min_spanning_tree(&complete));

    println!["Got {} points", points.len()];
    return (mst, points);
}

fn masked_maximin_4d_tree(
    intensities: &Array<f64, Ix4>,
    mask: &Array<u8, Ix3>,
) -> (
    Graph<(usize, usize, usize), (f64, f64), petgraph::Undirected>,
    Vec<(usize, usize, usize)>,
) {
    let n_dim = mask.ndim();
    let (c, z, y, x) = intensities.dim();
    let mut node_indices =
        Array::<usize, _>::from_elem(ndarray::Dim((z, y, x)), usize::max_value());
    let mut complete: Graph<(usize, usize, usize), (f64, f64), petgraph::Undirected> =
        Graph::new_undirected();

    let mut points = Vec::new();

    for i in 0..z {
        for j in 0..y {
            for k in 0..x {
                let index = (i, j, k);
                let index_slice = s![.., index.0, index.1, index.2];
                let value = intensities.slice(index_slice);
                let node_index = complete.add_node(index);

                if *mask
                    .get(index)
                    .expect(&format!["Mask does not have a value at {:?}", index])
                    > 0u8
                {
                    points.push(index);
                }
                node_indices[index] = node_index.index();

                for i in 0..n_dim {
                    let mut adj: Vec<usize> = (0..n_dim).map(|_| 0).collect();
                    adj[i] = 1;
                    let up_index = (
                        usize::wrapping_add(index.0, adj[0]),
                        usize::wrapping_add(index.1, adj[1]),
                        usize::wrapping_add(index.2, adj[2]),
                    );
                    let up_index_slice = s![.., up_index.0, up_index.1, up_index.2];
                    let down_index = (
                        usize::wrapping_sub(index.0, adj[0]),
                        usize::wrapping_sub(index.1, adj[1]),
                        usize::wrapping_sub(index.2, adj[2]),
                    );
                    let down_index_slice = s![.., down_index.0, down_index.1, down_index.2];
                    let up_index = node_indices.get(up_index);
                    let down_index = node_indices.get(down_index);
                    match up_index {
                        Some(up_index) => {
                            if up_index < &usize::max_value() {
                                let difference = &value - &intensities.slice(up_index_slice);
                                let dot_prod = difference.dot(&difference);
                                complete.add_edge(
                                    node_index,
                                    NodeIndex::new(*up_index),
                                    (f64::sqrt(dot_prod), 1.0),
                                );
                            }
                        }
                        None => (),
                    }
                    match down_index {
                        Some(down_index) => {
                            if down_index < &usize::max_value() {
                                let difference = &value - &intensities.slice(down_index_slice);
                                let dot_prod = difference.dot(&difference);
                                complete.add_edge(
                                    node_index,
                                    NodeIndex::new(*down_index),
                                    (f64::sqrt(dot_prod), 1.0),
                                );
                            }
                        }
                        None => (),
                    }
                }
            }
        }
    }

    let mst: Graph<(usize, usize, usize), (f64, f64), petgraph::Undirected> =
        Graph::from_elements(min_spanning_tree(&complete));

    println!["Got {} points", points.len()];
    return (mst, points);
}

fn tree_edges(
    graph: Graph<(usize, usize, usize), (f64, f64), petgraph::Undirected>,
) -> Vec<((usize, usize, usize), (usize, usize, usize))> {
    let mst_edges: Vec<((usize, usize, usize), (usize, usize, usize))> = min_spanning_tree(&graph)
        .filter_map(|elem| match elem {
            Element::Edge {
                source,
                target,
                weight: _,
            } => Some((
                *graph.node_weight(NodeIndex::new(source))?,
                *graph.node_weight(NodeIndex::new(target))?,
            )),
            Element::Node { weight: _ } => None,
        })
        .collect();
    return mst_edges;
}

fn into_3d<T: Clone>(intensities: ArrayViewD<T>) -> Result<ndarray::Array<T, Ix3>, PyErr> {
    let intensities = intensities.to_owned().into_dimensionality::<Ix3>();
    let intensities = match intensities {
        Err(e) => return Err(PyErr::from(ShapeErrorWrapper { err: e })),
        Ok(array) => array,
    };
    return Ok(intensities);
}

fn into_4d<T: Clone>(intensities: ArrayViewD<T>) -> Result<ndarray::Array<T, Ix4>, PyErr> {
    let intensities = intensities.to_owned().into_dimensionality::<Ix4>();
    let intensities = match intensities {
        Err(e) => return Err(PyErr::from(ShapeErrorWrapper { err: e })),
        Ok(array) => array,
    };
    return Ok(intensities);
}

#[pyfunction]
fn maximin_tree_edges(
    py: Python,
    intensities: &PyArrayDyn<f64>,
) -> PyResult<Vec<((usize, usize, usize), (usize, usize, usize))>> {
    let intensities = intensities.as_array();
    let intensities = into_3d(intensities)?;
    let tree = maximin_3d_tree(&intensities);
    Ok(tree_edges(tree))
}

pub fn dijkstra<G, F, K>(
    graph: G,
    start: G::NodeId,
    goal: Option<G::NodeId>,
    mut edge_cost: F,
    initial_value: K,
) -> HashMap<G::NodeId, K>
where
    G: IntoEdges + Visitable,
    G::NodeId: Eq + Hash,
    F: FnMut(G::EdgeRef, K) -> K,
    K: Measure + Copy,
{
    let mut visited = graph.visit_map();
    let mut scores = HashMap::new();
    //let mut predecessor = HashMap::new();
    let mut visit_next = BinaryHeap::new();
    let zero_score = initial_value;
    scores.insert(start, zero_score);
    visit_next.push(MinScored(zero_score, start));
    while let Some(MinScored(node_score, node)) = visit_next.pop() {
        if visited.is_visited(&node) {
            continue;
        }
        if goal.as_ref() == Some(&node) {
            break;
        }
        for edge in graph.edges(node) {
            let next = edge.target();
            if visited.is_visited(&next) {
                continue;
            }
            let mut next_score = edge_cost(edge, node_score);
            match scores.entry(next) {
                Occupied(ent) => {
                    if next_score < *ent.get() {
                        *ent.into_mut() = next_score;
                    //predecessor.insert(next.clone(), node.clone());
                    } else {
                        next_score = *ent.get();
                    }
                }
                Vacant(ent) => {
                    ent.insert(next_score);
                    //predecessor.insert(next.clone(), node.clone());
                }
            }
            visit_next.push(MinScored(next_score, next));
        }
        visited.visit(node);
    }
    scores
}

fn query_tree(
    tree: &Graph<(usize, usize, usize), (f64, f64), petgraph::Undirected>,
    intensities: &ndarray::Array<f64, Ix3>,
    query_tuples: &Vec<(usize, usize)>,
) -> Vec<f64> {
    let tree = tree.map(
        |n_ind, n| *n,
        |e_ind, e| {
            let (u, v) = tree.edge_endpoints(e_ind).expect("bad edge index");
            f64::min(
                *intensities
                    .get(*tree.node_weight(u).expect("bad node index"))
                    .expect("bad matrix index"),
                *intensities
                    .get(*tree.node_weight(v).expect("bad node index"))
                    .expect("bad matrix index"),
            )
        },
    );
    let mut scores = Vec::new();
    for (u, v) in query_tuples.iter() {
        let dijk = dijkstra(
            &tree,
            NodeIndex::new(*u),
            Some(NodeIndex::new(*v)),
            |w, c| f64::min(*w.weight(), c),
            1.0f64 / 0.0f64,
        );
        scores.push(
            *dijk
                .get(&NodeIndex::new(*v))
                .expect("No score for target vertex"),
        );
    }
    return scores;
}

fn last_occurance<T: std::cmp::Eq + std::hash::Hash>(vs: &Vec<T>, ps: HashSet<T>) -> Option<&T> {
    let mut last = vs.iter().rev();
    loop {
        match last.next() {
            Some(v) => {
                if ps.contains(v) {
                    return Some(v);
                }
            }
            None => return None,
        };
    }
}

fn reduce_tree(
    mut tree: Graph<(usize, usize, usize), (f64, f64), petgraph::Undirected>,
    query_points: Vec<(usize, usize, usize)>,
) -> Graph<(usize, usize, usize), (f64, f64), petgraph::Undirected> {
    let mut nodes_to_keep: HashSet<(usize, usize, usize)> = HashSet::new();
    let query_points: HashSet<(usize, usize, usize)> = query_points.into_iter().collect();

    let mut start = None;
    for node_index in tree.node_indices() {
        if query_points.contains(tree.node_weight(node_index).unwrap()) {
            start = Some(node_index);
        }
    }
    let mut dfs = Dfs::new(&tree, NodeIndex::new(start.unwrap().index()));
    let mut seen_nodes: Vec<NodeIndex> = vec![];

    let mut i = 0;
    while let Some(current) = dfs.next(&tree) {
        if i % 1000 == 0 {
            println!["traversed {} nodes!", i];
        }
        i += 1;

        if seen_nodes.len() == 0 {
            if !query_points.contains(tree.node_weight(current).unwrap()) {
                panic!["first node is not in query_points"];
            }
            seen_nodes.push(current);
            nodes_to_keep.insert(*tree.node_weight(current).unwrap());
            continue;
        }

        let current_id = current.index();

        // pop nodes that aren't neighbors of current.
        // Since this is a dfs, we must have come accross currents parents before,
        // so this is guaranteed to end before seen_nodes is empty.
        loop {
            let candidate = seen_nodes.pop().expect("no previously seen nodes!");
            match tree.find_edge(candidate, current) {
                Some(_) => {
                    seen_nodes.push(candidate);
                    break;
                }
                None => (),
            }
        }

        seen_nodes.push(current);

        // If the current id is a query point, push to query points and
        // add nodes up to previous query point to tree
        if query_points.contains(tree.node_weight(current).unwrap()) {
            let mut last = seen_nodes.iter().rev();
            loop {
                let candidate = last.next().expect("No more candidates in seen_nodes");
                if nodes_to_keep.contains(tree.node_weight(*candidate).unwrap()) {
                    break;
                } else {
                    nodes_to_keep.insert(*tree.node_weight(*candidate).unwrap());
                }
            }
        }
    }

    tree.retain_nodes(|tree, node| nodes_to_keep.contains(tree.node_weight(node).unwrap()));

    let mut nodes_to_collapse = vec![];

    for node in tree.node_indices() {
        if !query_points.contains(tree.node_weight(node).unwrap()) {
            nodes_to_collapse.push(node);
        }
    }

    nodes_to_collapse.sort_unstable();
    for node in nodes_to_collapse.iter().rev() {
        let mut neighbors = tree.neighbors(*node).detach();
        let (edge_a, node_a) = match neighbors.next(&tree) {
            Some(x) => x,
            None => {
                tree.remove_node(*node).unwrap();
                continue;
            }
        };
        let (edge_b, node_b) = match neighbors.next(&tree) {
            Some(x) => x,
            None => {
                tree.remove_node(*node).unwrap();
                continue;
            }
        };
        match neighbors.next(&tree) {
            Some(_) => continue,
            None => (),
        };

        let weight_a = *tree.edge_weight(edge_a).unwrap();
        let weight_b = *tree.edge_weight(edge_b).unwrap();
        tree.add_edge(
            node_a,
            node_b,
            (
                f64::max(weight_a.0, weight_b.0),
                f64::min(weight_a.1, weight_b.1),
            ),
        );
        tree.remove_edge(edge_a);
        tree.remove_edge(edge_b);
    }
    return tree;
}

/// given an python ndarray of intensities (f64) and mask (u8) return pairwise costs between
/// every voxel in the mask.
#[pyfunction]
fn maximin_tree_query(
    intensities: &PyArrayDyn<f64>,
    mask: &PyArrayDyn<u8>,
) -> PyResult<Vec<((usize, usize, usize), (usize, usize, usize), f64)>> {
    let intensities = into_3d(intensities.as_array())?;
    let mask = into_3d(mask.as_array())?;
    let (mut tree, query_points) = masked_maximin_3d_tree(&intensities, &mask);
    println![
        "Got tree with {} nodes and {} edges",
        tree.node_count(),
        tree.edge_count()
    ];
    let sub_tree = reduce_tree(tree, query_points);
    println!["Reduced tree to {} nodes", sub_tree.node_count()];
    let results: Vec<((usize, usize, usize), (usize, usize, usize), f64)> = sub_tree
        .edge_indices()
        .map(|e_ind| {
            let (u, v) = sub_tree.edge_endpoints(e_ind).unwrap();
            let (min_intensity, _max_intensity) = sub_tree.edge_weight(e_ind).unwrap();
            (
                *sub_tree
                    .node_weight(u)
                    .expect(&format!["Query node {:?} missing", u]),
                *sub_tree
                    .node_weight(v)
                    .expect(&format!["Query node {:?} missing", v]),
                -min_intensity,
            )
        })
        .collect();
    Ok(results)
}

/// given an python ndarray of intensities (f64) and mask (u8) return pairwise costs between
/// every voxel in the mask.
#[pyfunction]
fn maximin_tree_query_hd(
    intensities: &PyArrayDyn<f64>,
    mask: &PyArrayDyn<u8>,
) -> PyResult<Vec<((usize, usize, usize), (usize, usize, usize), f64)>> {
    let intensities = into_4d(intensities.as_array())?;
    println!["{:?}", intensities.dim()];
    let mask = into_3d(mask.as_array())?;
    println!["{:?}", mask.dim()];
    let (mut tree, query_points) = masked_maximin_4d_tree(&intensities, &mask);
    println![
        "Got tree with {} nodes and {} edges",
        tree.node_count(),
        tree.edge_count()
    ];
    let sub_tree = reduce_tree(tree, query_points);
    println!["Reduced tree to {} nodes", sub_tree.node_count()];
    let results: Vec<((usize, usize, usize), (usize, usize, usize), f64)> = sub_tree
        .edge_indices()
        .map(|e_ind| {
            let (u, v) = sub_tree.edge_endpoints(e_ind).unwrap();
            let (min_intensity, _max_intensity) = sub_tree.edge_weight(e_ind).unwrap();
            (
                *sub_tree
                    .node_weight(u)
                    .expect(&format!["Query node {:?} missing", u]),
                *sub_tree
                    .node_weight(v)
                    .expect(&format!["Query node {:?} missing", v]),
                *min_intensity,
            )
        })
        .collect();
    Ok(results)
}

/// This module is a python module implemented in Rust.
#[pymodule]
fn maximin(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(maximin_tree_edges))?;
    m.add_wrapped(wrap_pyfunction!(maximin_tree_query))?;
    m.add_wrapped(wrap_pyfunction!(maximin_tree_query_hd))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    #[test]
    fn test_maximin_tree() {
        let x = array![[[0.0, 1.0], [3.0, 2.0]], [[7.0, 6.0], [5.0, 4.0]]];
        let tree = maximin_3d_tree(&x);
        assert_eq![tree.edge_count(), 7];
    }

    fn test_maximin_tree_wrong_dim() {
        let x = array![[[0.0, 1.0], [3.0, 2.0]]].into_dyn();
        let y = x.view();
        let edges = into_3d(y).unwrap();
    }
}
