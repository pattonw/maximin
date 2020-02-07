

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