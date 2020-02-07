import numpy as np
import pytest

from maximin import maximin_tree_edges


@pytest.mark.skip("Not implemented yet")
def test_maximin_tree_large():
    intensities = np.array(
        [
            [1.0, 0.9, 0.8, 0.7, 0.6],
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.0, 0.0, 0.1, 0.2, 0.3],
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [1.0, 0.9, 0.8, 0.7, 0.6],
        ]
    )

    tree = maximin_tree_edges(intensities)

    expected_edges = ["ugh"]

    assert tree == expected_edges


def test_maximin_tree_2d():
    intensities = np.array([[0, 1], [3, 2]], dtype=float)

    with pytest.raises(Exception):
        tree = maximin_tree_edges(intensities)


def test_maximin_tree_array():
    intensities = np.array(
        [[[0.0, 1.0], [3.0, 2.0]], [[7.0, 6.0], [5.0, 4.0]]], dtype=float
    )

    tree = maximin_tree_edges(intensities)

    n_to_index = {
        (1, 0, 0): 7,
        (1, 0, 1): 6,
        (1, 1, 0): 5,
        (1, 1, 1): 4,
        (0, 1, 0): 3,
        (0, 1, 1): 2,
        (0, 0, 1): 1,
        (0, 0, 0): 0,
    }

    expected_edges = {
        7: [(7, 6), (7, 5), (7, 0)],
        6: [(6, 4), (6, 1)],
        5: [(5, 3)],
        4: [(4, 2)],
        3: [],
        2: [],
        1: [],
    }

    start = 0
    for i in range(7):
        k = 7 - i
        expected = expected_edges.pop(k)
        n = len(expected)
        seen = [(n_to_index[u], n_to_index[v]) for u, v in tree[start : start + n]]
        seen = [(u, v) if u > v else (v, u) for u, v in seen]
        seen = sorted(seen, key=lambda x: -x[1])
        start += n

        assert seen == expected

