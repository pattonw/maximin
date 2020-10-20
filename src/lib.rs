#![feature(test)]

extern crate test;
// use ndarray_rand;
// use rand_isaac;

#[macro_use]
use ndarray::ShapeError;
// use ndarray::linalg::Dot;
use ndarray::{array, s, Array, ArrayD, ArrayViewD, Ix3, Ix4, IxDynImpl};
use numpy::{IntoPyArray, PyArrayDyn};

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::{exceptions, PyErr, PyResult};

use petgraph::algo::min_spanning_tree;
use petgraph::data::*;
use petgraph::graph::NodeIndex;
use petgraph::stable_graph::StableGraph;
use petgraph::visit::Dfs;

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
        exceptions::PyOSError::new_err(err.to_string())
    }
}

fn maximin_3d_tree(
    intensities: &Array<f64, Ix3>,
    threshold: f64,
) -> StableGraph<(usize, usize, usize), (f64, f64), petgraph::Undirected> {
    let n_dim = intensities.ndim();
    let dims = intensities.raw_dim();
    let mut node_indices = Array::<usize, _>::from_elem(dims, usize::max_value());
    let mut complete: StableGraph<(usize, usize, usize), (f64, f64), petgraph::Undirected> =
        StableGraph::with_capacity(
            dims[0] * dims[1] * dims[2],
            2 * n_dim * dims[0] * dims[1] * dims[2],
        );

    let mut node_index_map: HashMap<usize, (usize, usize, usize)> = HashMap::new();

    for (index, value) in intensities.indexed_iter() {
        if value >= &threshold {
            let node_index = complete.add_node(index);
            node_index_map.insert(node_index.index(), index.clone());
            node_indices[index] = node_index.index();
            for i in 0..n_dim {
                let mut adj: Vec<usize> = (0..n_dim).map(|_| 0).collect();
                adj[i] = 1;
                let down_neighbor_index = (
                    usize::wrapping_sub(index.0, adj[0]),
                    usize::wrapping_sub(index.1, adj[1]),
                    usize::wrapping_sub(index.2, adj[2]),
                );

                let down_index = node_indices.get(down_neighbor_index);

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
    }

    return complete;
}

fn mst(
    complete: StableGraph<(usize, usize, usize), (f64, f64), petgraph::Undirected>,
) -> StableGraph<(usize, usize, usize), (f64, f64), petgraph::Undirected> {
    let mst: StableGraph<(usize, usize, usize), (f64, f64), petgraph::Undirected> =
        StableGraph::from_elements(min_spanning_tree(&complete));
    return mst;
}

fn masked_maximin_3d_tree(
    intensities: &Array<f64, Ix3>,
    mask: &Array<u8, Ix3>,
    threshold: f64,
) -> (
    StableGraph<(usize, usize, usize), (f64, f64), petgraph::Undirected>,
    HashSet<NodeIndex>,
) {
    let n_dim = intensities.ndim();
    let dims = mask.raw_dim();
    let mut node_indices = Array::<usize, _>::from_elem(intensities.raw_dim(), usize::max_value());
    let mut complete: StableGraph<(usize, usize, usize), (f64, f64), petgraph::Undirected> =
        StableGraph::with_capacity(
            dims[0] * dims[1] * dims[2],
            2 * n_dim * dims[0] * dims[1] * dims[2],
        );

    let mut critical_points = HashSet::new();

    for (index, candidate) in mask.indexed_iter() {
        let value = intensities[index];

        if value >= threshold {
            let node_index = complete.add_node(index);
            node_indices[index] = node_index.index();

            if candidate > &0u8 {
                critical_points.insert(node_index);
            }
            for i in 0..n_dim {
                let mut adj: Vec<usize> = (0..n_dim).map(|_| 0).collect();
                adj[i] = 1;
                let down_neighbor_index = (
                    usize::wrapping_sub(index.0, adj[0]),
                    usize::wrapping_sub(index.1, adj[1]),
                    usize::wrapping_sub(index.2, adj[2]),
                );

                let down_index = node_indices.get(down_neighbor_index);

                // either down index exists or that node was skipped due to threshold
                match down_index {
                    Some(down_index) => {
                        if down_index < &usize::max_value() {
                            let min_intensity = f64::min(value, intensities[down_neighbor_index]);
                            let max_intensity = f64::max(value, intensities[down_neighbor_index]);
                            if min_intensity >= threshold {
                                complete.add_edge(
                                    node_index,
                                    NodeIndex::new(*down_index),
                                    (-min_intensity, -max_intensity),
                                );
                            }
                        }
                    }
                    None => (),
                }
            }
        } else {
            // keep candidates even if they are below threshold.
            if candidate > &0u8 {
                let node_index = complete.add_node(index);
                node_indices[index] = node_index.index();
                critical_points.insert(node_index);
            }
        }
    }

    return (complete, critical_points);
}

fn masked_maximin_4d_tree(
    intensities: &Array<f64, Ix4>,
    mask: &Array<u8, Ix3>,
) -> (
    StableGraph<(usize, usize, usize), (f64, f64), petgraph::Undirected>,
    HashSet<NodeIndex>,
) {
    let n_dim = mask.ndim();
    let dims = mask.raw_dim();
    let mut node_indices = Array::<usize, _>::from_elem(dims, usize::max_value());
    let mut complete: StableGraph<(usize, usize, usize), (f64, f64), petgraph::Undirected> =
        StableGraph::with_capacity(
            dims[0] * dims[1] * dims[2],
            2 * n_dim * dims[0] * dims[1] * dims[2],
        );

    let mut critical_points = HashSet::new();

    for (index, candidate) in mask.indexed_iter() {
        let node_index = complete.add_node(index);

        if candidate > &0u8 {
            critical_points.insert(node_index);
        }
        let index_slice = s![.., index.0, index.1, index.2];
        let value = intensities.slice(index_slice);

        node_indices[index] = node_index.index();

        for i in 0..n_dim {
            let mut adj: Vec<usize> = (0..n_dim).map(|_| 0).collect();
            adj[i] = 1;
            let down_index = (
                usize::wrapping_sub(index.0, adj[0]),
                usize::wrapping_sub(index.1, adj[1]),
                usize::wrapping_sub(index.2, adj[2]),
            );
            let down_index_slice = s![.., down_index.0, down_index.1, down_index.2];
            let down_index = node_indices.get(down_index);
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
    return (complete, critical_points);
}

fn tree_edges(
    graph: StableGraph<(usize, usize, usize), (f64, f64), petgraph::Undirected>,
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

/// Get all edges of a minimum spanning tree over the given intensities.
/// Edges returned as index pairs (a, b) where a and b are array indices.
#[pyfunction(threshold = "0.0")]
fn maximin_tree_edges(
    _py: Python,
    intensities: &PyArrayDyn<f64>,
    threshold: f64,
) -> PyResult<Vec<((usize, usize, usize), (usize, usize, usize))>> {
    let intensities = unsafe { intensities.as_array() };
    let intensities = into_3d(intensities)?;
    let complete = maximin_3d_tree(&intensities, threshold);
    let tree = mst(complete);
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

/// A method for querying the maximum intensity between two nodes on a tree
fn query_tree(
    tree: &StableGraph<(usize, usize, usize), (f64, f64), petgraph::Undirected>,
    intensities: &ndarray::Array<f64, Ix3>,
    query_tuples: &Vec<(usize, usize)>,
) -> Vec<f64> {
    let tree = tree.map(
        |_n_ind, n| *n,
        |e_ind, _e| {
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

/// Remove excess nodes and edges from the mst. For every node in the tree, if
/// removing the node would not cause any critical points to land on seperate
/// connected components, that node is removed along with all adjacent edges.
fn trim_mst(
    mut tree: StableGraph<(usize, usize, usize), (f64, f64), petgraph::Undirected>,
    critical_points: &HashSet<NodeIndex>,
) -> StableGraph<(usize, usize, usize), (f64, f64), petgraph::Undirected> {
    let mut subtree_nodes: HashSet<NodeIndex> = HashSet::new();

    let mut to_visit = critical_points.clone();

    while to_visit.len() > 0 {
        let start: &NodeIndex = to_visit.iter().next().unwrap();
        let mut dfs = Dfs::new(&tree, *start);
        let mut seen_nodes: Vec<NodeIndex> = vec![];
        while let Some(current) = dfs.next(&tree) {
            if seen_nodes.len() == 0 {
                assert!(
                    to_visit.remove(&current),
                    "current {:?} wasnt in to_visit: {:?}",
                    current,
                    to_visit
                );
                seen_nodes.push(current);
                subtree_nodes.insert(current);
                continue;
            }

            // backtrack up dfs until a neighbor of current is found
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
            // add nodes up to previously seen point to tree
            if to_visit.contains(&current) {
                to_visit.remove(&current);
                let mut last = seen_nodes.iter().rev();
                loop {
                    let candidate = last.next().expect("No more candidates in seen_nodes");
                    if subtree_nodes.contains(candidate) {
                        break;
                    } else {
                        subtree_nodes.insert(*candidate);
                    }
                }
            }
        }
    }

    tree.retain_nodes(|_tree, node| subtree_nodes.contains(&node));

    return tree;
}

/// Collapse chains of nodes between critical points to a single edge with score equal
/// to the maximum edge between the critical points.
fn decimate_mst(
    mut tree: StableGraph<(usize, usize, usize), (f64, f64), petgraph::Undirected>,
    critical_points: &HashSet<NodeIndex>,
) -> StableGraph<(usize, usize, usize), (f64, f64), petgraph::Undirected> {
    let mut nodes_to_collapse = vec![];

    for node in tree.node_indices() {
        if !critical_points.contains(&node) {
            nodes_to_collapse.push(node);
        }
    }

    nodes_to_collapse.sort_unstable();
    for node in nodes_to_collapse.iter().rev() {
        let mut neighbors = tree.neighbors(*node).detach();
        let (edge_a, node_a) = match neighbors.next(&tree) {
            Some(x) => x,
            None => {
                panic![format!["node {:?} has no neighbors", node]];
            }
        };
        let (edge_b, node_b) = match neighbors.next(&tree) {
            Some(x) => x,
            None => {
                panic![format!["node {:?} has only 1 neighbor", node]];
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

fn tree_to_edges(
    tree: StableGraph<(usize, usize, usize), (f64, f64), petgraph::Undirected>,
    negate: bool,
) -> Vec<((usize, usize, usize), (usize, usize, usize), f64)> {
    tree.edge_indices()
        .map(|e_ind| {
            let (u, v) = tree.edge_endpoints(e_ind).unwrap();
            let (mut min_intensity, _max_intensity) = tree.edge_weight(e_ind).unwrap();
            if negate {
                min_intensity = -min_intensity;
            }
            (
                *tree
                    .node_weight(u)
                    .expect(&format!["Query node {:?} missing", u]),
                *tree
                    .node_weight(v)
                    .expect(&format!["Query node {:?} missing", v]),
                min_intensity,
            )
        })
        .collect()
}

#[pyfunction(decimate = "true", threshold = "0.0")]
fn maximin_tree_query(
    intensities: &PyArrayDyn<f64>,
    mask: &PyArrayDyn<u8>,
    decimate: bool,
    threshold: f64,
) -> PyResult<Vec<((usize, usize, usize), (usize, usize, usize), f64)>> {
    let intensities = into_3d(unsafe { intensities.as_array() })?;
    let mask = into_3d(unsafe { mask.as_array() })?;
    let (complete, critical_points) = masked_maximin_3d_tree(&intensities, &mask, threshold);
    let tree = mst(complete);
    let mut sub_tree = trim_mst(tree, &critical_points);
    if decimate {
        sub_tree = decimate_mst(sub_tree, &critical_points);
    }
    let results = tree_to_edges(sub_tree, true);
    Ok(results)
}
#[pyfunction(threshold = "0.0")]
fn maximin_tree_query_plus_decimated(
    intensities: &PyArrayDyn<f64>,
    mask: &PyArrayDyn<u8>,
    threshold: f64,
) -> PyResult<(
    Vec<((usize, usize, usize), (usize, usize, usize), f64)>,
    Vec<((usize, usize, usize), (usize, usize, usize), f64)>,
)> {
    let intensities = into_3d(unsafe { intensities.as_array() })?;
    let mask = into_3d(unsafe { mask.as_array() })?;
    let (complete, critical_points) = masked_maximin_3d_tree(&intensities, &mask, threshold);
    let tree = mst(complete);
    let sub_tree = trim_mst(tree, &critical_points);
    let decimated_sub_tree = decimate_mst(sub_tree.clone(), &critical_points);
    let results = tree_to_edges(sub_tree, true);
    let decimated_results = tree_to_edges(decimated_sub_tree, true);
    Ok((results, decimated_results))
}

#[pyfunction(decimate = "true")]
fn maximin_tree_query_hd(
    intensities: &PyArrayDyn<f64>,
    mask: &PyArrayDyn<u8>,
    decimate: bool,
) -> PyResult<Vec<((usize, usize, usize), (usize, usize, usize), f64)>> {
    let intensities = into_4d(unsafe { intensities.as_array() })?;
    let mask = into_3d(unsafe { mask.as_array() })?;
    let (complete, critical_points) = masked_maximin_4d_tree(&intensities, &mask);
    let tree = mst(complete);
    let mut sub_tree = trim_mst(tree, &critical_points);
    if decimate {
        sub_tree = decimate_mst(sub_tree, &critical_points);
    }
    let results = tree_to_edges(sub_tree, false);
    Ok(results)
}
#[pyfunction]
fn maximin_tree_query_hd_plus_decimated(
    intensities: &PyArrayDyn<f64>,
    mask: &PyArrayDyn<u8>,
) -> PyResult<(
    Vec<((usize, usize, usize), (usize, usize, usize), f64)>,
    Vec<((usize, usize, usize), (usize, usize, usize), f64)>,
)> {
    let intensities = into_4d(unsafe { intensities.as_array() })?;
    let mask = into_3d(unsafe { mask.as_array() })?;
    let (complete, critical_points) = masked_maximin_4d_tree(&intensities, &mask);
    let tree = mst(complete);
    let sub_tree = trim_mst(tree, &critical_points);
    let decimated_sub_tree = decimate_mst(sub_tree.clone(), &critical_points);
    let results = tree_to_edges(sub_tree, false);
    let decimated_results = tree_to_edges(decimated_sub_tree, false);
    Ok((results, decimated_results))
}

#[pymodule]
fn maximin(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(maximin_tree_edges))?;
    m.add_wrapped(wrap_pyfunction!(maximin_tree_query))?;
    m.add_wrapped(wrap_pyfunction!(maximin_tree_query_plus_decimated))?;
    m.add_wrapped(wrap_pyfunction!(maximin_tree_query_hd))?;
    m.add_wrapped(wrap_pyfunction!(maximin_tree_query_hd_plus_decimated))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use ndarray::Array;
    use ndarray_rand::rand::SeedableRng;
    use ndarray_rand::rand_distr::Binomial;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;
    use rand_isaac::isaac64::Isaac64Rng;
    use test::Bencher;

    static BENCH_SIZE: usize = 20;

    #[test]
    fn test_maximin_tree() {
        let x = array![[[0.0, 1.0], [3.0, 2.0]], [[7.0, 6.0], [5.0, 4.0]]];
        let complete = maximin_3d_tree(&x, 0.0);
        let tree = mst(complete);
        assert_eq![tree.edge_count(), 7];
    }
    #[bench]
    fn bench_maximin_tree(b: &mut Bencher) {
        let x = array![[[0.0, 1.0], [3.0, 2.0]], [[7.0, 6.0], [5.0, 4.0]]];
        b.iter(|| maximin_3d_tree(&x, 0.0));
    }
    #[bench]
    fn bench_random_maximin_tree(b: &mut Bencher) {
        // Get a seeded random number generator for reproducibility (Isaac64 algorithm)
        let seed = 42;
        let mut rng = Isaac64Rng::seed_from_u64(seed);

        // Generate a random array using `rng`
        let intensities = Array::random_using(
            (BENCH_SIZE, BENCH_SIZE, BENCH_SIZE),
            Uniform::new(0., 1.0),
            &mut rng,
        );
        let mask = Array::random_using(
            (BENCH_SIZE, BENCH_SIZE, BENCH_SIZE),
            Uniform::new(0., 1.0),
            &mut rng,
        )
        .mapv(|a| (a > 0.5).into());
        b.iter(|| masked_maximin_3d_tree(&intensities, &mask, 0.0));
    }
    #[bench]
    fn bench_random_mst(b: &mut Bencher) {
        // Get a seeded random number generator for reproducibility (Isaac64 algorithm)
        let seed = 42;
        let mut rng = Isaac64Rng::seed_from_u64(seed);

        // Generate a random array using `rng`
        let intensities = Array::random_using(
            (BENCH_SIZE, BENCH_SIZE, BENCH_SIZE),
            Uniform::new(0., 1.0),
            &mut rng,
        );
        let mask = Array::random_using(
            (BENCH_SIZE, BENCH_SIZE, BENCH_SIZE),
            Uniform::new(0., 1.0),
            &mut rng,
        )
        .mapv(|a| (a > 0.5).into());
        let (complete, _critical_points) = masked_maximin_3d_tree(&intensities, &mask, 0.0);
        b.iter(|| mst(complete.clone()));
    }
    #[bench]
    fn bench_random_mst_thresholded(b: &mut Bencher) {
        // Get a seeded random number generator for reproducibility (Isaac64 algorithm)
        let seed = 42;
        let mut rng = Isaac64Rng::seed_from_u64(seed);

        // Generate a random array using `rng`
        let intensities = Array::random_using(
            (BENCH_SIZE, BENCH_SIZE, BENCH_SIZE),
            Uniform::new(0., 1.0),
            &mut rng,
        );
        let mask = Array::random_using(
            (BENCH_SIZE, BENCH_SIZE, BENCH_SIZE),
            Uniform::new(0., 1.0),
            &mut rng,
        )
        .mapv(|a| (a > 0.5).into());
        let (complete, _critical_points) = masked_maximin_3d_tree(&intensities, &mask, 0.5);
        b.iter(|| mst(complete.clone()));
    }
    #[bench]
    fn bench_random_trim_tree(b: &mut Bencher) {
        // Get a seeded random number generator for reproducibility (Isaac64 algorithm)
        let seed = 42;
        let mut rng = Isaac64Rng::seed_from_u64(seed);

        // Generate a random array using `rng`
        let intensities = Array::random_using(
            (BENCH_SIZE, BENCH_SIZE, BENCH_SIZE),
            Uniform::new(0., 1.0),
            &mut rng,
        );
        let mask = Array::random_using(
            (BENCH_SIZE, BENCH_SIZE, BENCH_SIZE),
            Uniform::new(0., 1.0),
            &mut rng,
        )
        .mapv(|a| (a > 0.5).into());

        let (complete, critical_points) = masked_maximin_3d_tree(&intensities, &mask, 0.0);
        let tree = mst(complete);
        b.iter(|| trim_mst(tree.clone(), &critical_points));
    }
    #[bench]
    fn bench_random_decimate_tree(b: &mut Bencher) {
        // Get a seeded random number generator for reproducibility (Isaac64 algorithm)
        let seed = 42;
        let mut rng = Isaac64Rng::seed_from_u64(seed);

        // Generate a random array using `rng`
        let intensities = Array::random_using(
            (BENCH_SIZE, BENCH_SIZE, BENCH_SIZE),
            Uniform::new(0., 1.0),
            &mut rng,
        );
        let mask = Array::random_using(
            (BENCH_SIZE, BENCH_SIZE, BENCH_SIZE),
            Uniform::new(0., 1.0),
            &mut rng,
        )
        .mapv(|a| (a > 0.5).into());

        let (complete, critical_points) = masked_maximin_3d_tree(&intensities, &mask, 0.0);
        let tree = mst(complete);
        let sub_tree = trim_mst(tree, &critical_points);
        b.iter(|| decimate_mst(sub_tree.clone(), &critical_points));
    }
}
