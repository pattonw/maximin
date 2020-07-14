#![feature(test)]

extern crate test;
use ndarray_rand;
use rand_isaac;

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
        exceptions::OSError::py_err(err.to_string())
    }
}

fn maximin_3d_tree(
    intensities: &Array<f64, Ix3>,
) -> StableGraph<(usize, usize, usize), (f64, f64), petgraph::Undirected> {
    let n_dim = intensities.ndim();
    let mut node_indices = Array::<usize, _>::from_elem(intensities.raw_dim(), usize::max_value());
    let mut complete: StableGraph<(usize, usize, usize), (f64, f64), petgraph::Undirected> =
        StableGraph::default();

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

    let mst: StableGraph<(usize, usize, usize), (f64, f64), petgraph::Undirected> =
        StableGraph::from_elements(min_spanning_tree(&complete));
    return mst;
}

fn masked_maximin_3d_tree(
    intensities: &Array<f64, Ix3>,
    mask: &Array<u8, Ix3>,
) -> (
    StableGraph<(usize, usize, usize), (f64, f64), petgraph::Undirected>,
    HashSet<NodeIndex>,
) {
    let n_dim = intensities.ndim();
    let mut node_indices = Array::<usize, _>::from_elem(intensities.raw_dim(), usize::max_value());
    let mut complete: StableGraph<(usize, usize, usize), (f64, f64), petgraph::Undirected> =
        StableGraph::default();

    let mut critical_points = HashSet::new();

    for (index, candidate) in mask.indexed_iter() {
        let node_index = complete.add_node(index);

        if candidate > &0u8 {
            critical_points.insert(node_index);
        }

        let value = intensities
            .get(index)
            .expect(&format!["No intensity value for {:?}", index]);

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

    let mst: StableGraph<(usize, usize, usize), (f64, f64), petgraph::Undirected> =
        StableGraph::from_elements(min_spanning_tree(&complete));

    return (mst, critical_points);
}

fn masked_maximin_4d_tree(
    intensities: &Array<f64, Ix4>,
    mask: &Array<u8, Ix3>,
) -> (
    StableGraph<(usize, usize, usize), (f64, f64), petgraph::Undirected>,
    HashSet<NodeIndex>,
) {
    let n_dim = mask.ndim();
    let (_c, z, y, x) = intensities.dim();
    let mut node_indices =
        Array::<usize, _>::from_elem(ndarray::Dim((z, y, x)), usize::max_value());
    let mut complete: StableGraph<(usize, usize, usize), (f64, f64), petgraph::Undirected> =
        StableGraph::default();

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

    let mst: StableGraph<(usize, usize, usize), (f64, f64), petgraph::Undirected> =
        StableGraph::from_elements(min_spanning_tree(&complete));

    return (mst, critical_points);
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

#[pyfunction]
fn maximin_tree_edges(
    _py: Python,
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
    tree: &StableGraph<(usize, usize, usize), (f64, f64), petgraph::Undirected>,
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

fn trim_mst(
    mut tree: StableGraph<(usize, usize, usize), (f64, f64), petgraph::Undirected>,
    critical_points: &HashSet<NodeIndex>,
) -> StableGraph<(usize, usize, usize), (f64, f64), petgraph::Undirected> {
    let mut subtree_nodes: HashSet<NodeIndex> = HashSet::new();

    let start: &NodeIndex = critical_points.iter().next().unwrap();
    let mut dfs = Dfs::new(&tree, *start);
    let mut seen_nodes: Vec<NodeIndex> = vec![];

    while let Some(current) = dfs.next(&tree) {
        if seen_nodes.len() == 0 {
            if !critical_points.contains(&current) {
                panic!["first node is not in critical_points"];
            }
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
        // add nodes up to previous query point to tree
        if critical_points.contains(&current) {
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

    tree.retain_nodes(|_tree, node| subtree_nodes.contains(&node));

    return tree;
}

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

#[pyfunction(decimate = "true")]
fn maximin_tree_query(
    intensities: &PyArrayDyn<f64>,
    mask: &PyArrayDyn<u8>,
    decimate: bool,
) -> PyResult<Vec<((usize, usize, usize), (usize, usize, usize), f64)>> {
    let intensities = into_3d(intensities.as_array())?;
    let mask = into_3d(mask.as_array())?;
    let (tree, critical_points) = masked_maximin_3d_tree(&intensities, &mask);
    let mut sub_tree = trim_mst(tree, &critical_points);
    if decimate {
        sub_tree = decimate_mst(sub_tree, &critical_points);
    }
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

#[pyfunction(decimate = "true")]
fn maximin_tree_query_hd(
    intensities: &PyArrayDyn<f64>,
    mask: &PyArrayDyn<u8>,
    decimate: bool,
) -> PyResult<Vec<((usize, usize, usize), (usize, usize, usize), f64)>> {
    let intensities = into_4d(intensities.as_array())?;
    let mask = into_3d(mask.as_array())?;
    let (tree, critical_points) = masked_maximin_4d_tree(&intensities, &mask);
    let mut sub_tree = trim_mst(tree, &critical_points);
    if decimate {
        sub_tree = decimate_mst(sub_tree, &critical_points);
    }
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

#[pymodule]
fn maximin(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(maximin_tree_edges))?;
    m.add_wrapped(wrap_pyfunction!(maximin_tree_query))?;
    m.add_wrapped(wrap_pyfunction!(maximin_tree_query_hd))?;

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

    #[test]
    fn test_maximin_tree() {
        let x = array![[[0.0, 1.0], [3.0, 2.0]], [[7.0, 6.0], [5.0, 4.0]]];
        let tree = maximin_3d_tree(&x);
        assert_eq![tree.edge_count(), 7];
    }
    #[bench]
    fn bench_maximin_tree(b: &mut Bencher) {
        let x = array![[[0.0, 1.0], [3.0, 2.0]], [[7.0, 6.0], [5.0, 4.0]]];
        b.iter(|| maximin_3d_tree(&x));
    }
    #[bench]
    fn bench_random_maximin_tree(b: &mut Bencher) {
        // Get a seeded random number generator for reproducibility (Isaac64 algorithm)
        let seed = 42;
        let mut rng = Isaac64Rng::seed_from_u64(seed);

        // Generate a random array using `rng`
        let a = Array::random_using((10, 10, 10), Uniform::new(0., 10.), &mut rng);
        b.iter(|| maximin_3d_tree(&a));
    }
    #[bench]
    fn bench_random_trim_tree(b: &mut Bencher) {
        // Get a seeded random number generator for reproducibility (Isaac64 algorithm)
        let seed = 42;
        let mut rng = Isaac64Rng::seed_from_u64(seed);

        // Generate a random array using `rng`
        let intensities = Array::random_using((10, 10, 10), Uniform::new(0., 10.), &mut rng);
        let mask = Array::random_using((10, 10, 10), Uniform::new(0., 1.0), &mut rng)
            .mapv(|a| (a > 0.5).into());

        let (tree, critical_points) = masked_maximin_3d_tree(&intensities, &mask);
        b.iter(|| trim_mst(tree.clone(), &critical_points));
    }
    #[bench]
    fn bench_random_decimate_tree(b: &mut Bencher) {
        // Get a seeded random number generator for reproducibility (Isaac64 algorithm)
        let seed = 42;
        let mut rng = Isaac64Rng::seed_from_u64(seed);

        // Generate a random array using `rng`
        let intensities = Array::random_using((10, 10, 10), Uniform::new(0., 10.), &mut rng);
        let mask = Array::random_using((10, 10, 10), Uniform::new(0., 1.0), &mut rng)
            .mapv(|a| (a > 0.5).into());

        let (tree, critical_points) = masked_maximin_3d_tree(&intensities, &mask);
        let sub_tree = trim_mst(tree, &critical_points);
        b.iter(|| decimate_mst(sub_tree.clone(), &critical_points));
    }
}
