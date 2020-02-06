use ndarray::ShapeError;
use ndarray::{Array, ArrayD, ArrayViewD, ArrayViewMutD, Dim, Ix3, IxDynImpl};
use numpy::{IntoPyArray, PyArrayDyn};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::{exceptions, PyErr, PyResult};
use std::collections::HashMap;

use petgraph::algo::min_spanning_tree;
use petgraph::data::Element;
use petgraph::graph::NodeIndex;
use petgraph::Graph;

use std;

#[pyfunction]
fn get_42() -> PyResult<usize> {
    Ok(42)
}

fn rusty_double(x: ArrayViewD<f64>) -> ArrayD<f64> {
    2.0 * &x
}

#[pyfunction]
fn double(py: Python, x: &PyArrayDyn<f64>) -> Py<PyArrayDyn<f64>> {
    let x = x.as_array();
    let result = rusty_double(x);
    result.into_pyarray(py).to_owned()
}

fn rusty_maximin_tree_nd(intensities: ArrayViewD<f64>) -> () {
    let n_dim = intensities.ndim();
    let mut node_indices = Array::<usize, _>::from_elem(intensities.raw_dim(), usize::max_value());
    let mut complete: Graph<f64, f64, petgraph::Undirected> = Graph::new_undirected();

    let mut node_index_map: HashMap<usize, Dim<IxDynImpl>> = HashMap::new();

    for (index, value) in intensities.indexed_iter() {
        let node_index = complete.add_node(*value);
        node_index_map.insert(node_index.index(), index.clone());
        node_indices[&index] = node_index.index();
        for i in 0..n_dim {
            let mut adjacent_vector: Vec<usize> = (0..n_dim).map(|i| 0).collect();
            adjacent_vector[i] = 1;
            let up_neighbor_index = Dim(adjacent_vector.clone()) + index.clone();
            let down_neighbor_index = Dim(adjacent_vector.clone()) + index.clone();

            let up_index = node_indices.get(&up_neighbor_index);
            let down_index = node_indices.get(&down_neighbor_index);

            match up_index {
                Some(up_index) => {
                    if up_index < &usize::max_value() {
                        let max_intensity = f64::max(*value, intensities[&up_neighbor_index]);
                        complete.add_edge(node_index, NodeIndex::new(*up_index), -max_intensity);
                    }
                }
                None => (),
            }

            match down_index {
                Some(down_index) => {
                    if down_index < &usize::max_value() {
                        let max_intensity = f64::max(*value, intensities[&down_neighbor_index]);
                        complete.add_edge(node_index, NodeIndex::new(*down_index), -max_intensity);
                    }
                }
                None => (),
            }
        }
    }

    let mst_edges: Vec<(usize, usize)> = min_spanning_tree(&complete)
        .map(|elem| match elem {
            Element::Node { weight: _ } => unreachable![],
            Element::Edge {
                source,
                target,
                weight: _,
            } => (source, target),
        })
        .collect();
    // How to convert these indices into tuples or vecs of usize?
    let mst_edges = mst_edges.into_iter().map(|(u, v)| {
        (
            node_index_map
                .get(&u)
                .expect(&format!["Node {} does not have an array index", u]),
            node_index_map
                .get(&v)
                .expect(&format!["Node {} does not have an array index", v]),
        )
    });
}

fn rusty_maximin_tree_3d(
    intensities: Array<f64, Ix3>,
) -> Vec<((usize, usize, usize), (usize, usize, usize))> {
    let n_dim = intensities.ndim();
    let mut node_indices = Array::<usize, _>::from_elem(intensities.raw_dim(), usize::max_value());
    let mut complete: Graph<f64, f64, petgraph::Undirected> = Graph::new_undirected();

    let mut node_index_map: HashMap<usize, (usize, usize, usize)> = HashMap::new();

    for (index, value) in intensities.indexed_iter() {
        let node_index = complete.add_node(*value);
        node_index_map.insert(node_index.index(), index.clone());
        node_indices[index] = node_index.index();
        for i in 0..n_dim {
            let mut adj: Vec<usize> = (0..n_dim).map(|i| 0).collect();
            adj[i] = 1;
            let up_neighbor_index = (index.0 + adj[0], index.1 + adj[0], index.2 + adj[0]);
            let down_neighbor_index = (index.0 - adj[0], index.1 - adj[0], index.2 - adj[0]);

            let up_index = node_indices.get(up_neighbor_index);
            let down_index = node_indices.get(down_neighbor_index);

            match up_index {
                Some(up_index) => {
                    if up_index < &usize::max_value() {
                        let max_intensity = f64::max(*value, intensities[up_neighbor_index]);
                        complete.add_edge(node_index, NodeIndex::new(*up_index), -max_intensity);
                    }
                }
                None => (),
            }

            match down_index {
                Some(down_index) => {
                    if down_index < &usize::max_value() {
                        let max_intensity = f64::max(*value, intensities[down_neighbor_index]);
                        complete.add_edge(node_index, NodeIndex::new(*down_index), -max_intensity);
                    }
                }
                None => (),
            }
        }
    }

    let mst_edges: Vec<(usize, usize)> = min_spanning_tree(&complete)
        .map(|elem| match elem {
            Element::Node { weight: _ } => unreachable![],
            Element::Edge {
                source,
                target,
                weight: _,
            } => (source, target),
        })
        .collect();
    let mst_edges: Vec<((usize, usize, usize), (usize, usize, usize))> = mst_edges
        .into_iter()
        .map(|(u, v)| {
            (
                *node_index_map
                    .get(&u)
                    .expect(&format!["Node {} does not have an array index", u]),
                *node_index_map
                    .get(&v)
                    .expect(&format!["Node {} does not have an array index", v]),
            )
        })
        .collect();
    return mst_edges;
}

use std::error;
use std::fmt;

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
        self.source()
    }
}

impl std::convert::From<ShapeErrorWrapper> for PyErr {
    fn from(err: ShapeErrorWrapper) -> PyErr {
        exceptions::OSError::py_err(err.to_string())
    }
}

#[pyfunction]
fn maximin_tree(
    py: Python,
    intensities: &PyArrayDyn<f64>,
) -> PyResult<Vec<((usize, usize, usize), (usize, usize, usize))>> {
    let intensities = intensities
        .as_array()
        .to_owned()
        .into_dimensionality::<Ix3>();
    let intensities = match intensities {
        Err(e) => return Err(PyErr::from(ShapeErrorWrapper { err: e })),
        Ok(array) => array,
    };
    let tree = rusty_maximin_tree_3d(intensities);
    Ok(tree)
}

/// This module is a python module implemented in Rust.
#[pymodule]
fn maximin(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(get_42))?;
    m.add_wrapped(wrap_pyfunction!(double))?;
    m.add_wrapped(wrap_pyfunction!(maximin_tree))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn double() {
        let x = array![[1.0, 2.0], [3.0, 4.0]].into_dyn();
        let y = array![[2.0, 4.0], [6.0, 8.0]].into_dyn();
        let doubled_x = rusty_double(x.into_dyn().view()).to_owned();
        assert_eq!(doubled_x, y);
    }
    fn test_maximin_tree() {
        let x = array![[[0.0, 1.0], [3.0, 2.0]]];
        let edges = rusty_maximin_tree_3d(x);
        assert_eq!(
            edges,
            vec![
                ((0, 0, 0), (0, 1, 0)),
                ((0, 1, 0), (1, 1, 0)),
                ((1, 1, 0), (1, 0, 0))
            ]
        )
    }
}
