=========
minimax
=========

A Library for calculating the minimax cost between points in numpy array of intensity values.
First creates a graph G from the ndarray where each voxel is a node with edges between each adjacent voxel (no diagonals).
Then calculates a minimum spanning tree accross G s.t. for every pair of nodes (u, v) in G, the minimax
cost between (u, v) is simply the minimum intensity of all nodes passed through between (u, v).

Usage: for returning the entire mst use:
```py
intensities = np.array(
    [[[0.0, 1.0], [3.0, 2.0]], [[7.0, 6.0], [5.0, 4.0]]],
    dtype=float,
)

tree = maximin_tree_edges(intensities)
```
For returning the dense pairwise costs between a subset of pixels:
```py
intensities = np.array(
    [[[7.0, 6.0], [5.0, 4.0]], [[0.0, 1.0], [3.0, 2.0]], [[7.0, 6.0], [5.0, 4.0]]],
    dtype=float,
)
mask = np.array([[[0, 1], [0, 0]], [[0, 0], [1, 0]], [[1, 0], [0, 1]]], dtype=np.uint8)

costs = maximin_tree_query(intensities, mask)
``


* Free software: MIT license