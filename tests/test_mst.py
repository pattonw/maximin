import numpy as np
import pytest

from maximin import maximin_tree_edges


def test_maximin_tree_2d():
    intensities = np.array([[0, 1], [3, 2]], dtype=float)

    with pytest.raises(Exception):
        maximin_tree_edges(intensities)


def test_maximin_tree_simple():
    intensities = np.array(
        #    0    1      2    3        4    5      6    7
        [[[0.0, 1.0], [3.0, 2.0]], [[7.0, 6.0], [5.0, 4.0]]],
        dtype=float,
    )

    tree = maximin_tree_edges(intensities)

    n_to_index = {
        (0, 0, 0): 0,
        (0, 0, 1): 1,
        (0, 1, 0): 2,
        (0, 1, 1): 3,
        (1, 0, 0): 4,
        (1, 0, 1): 5,
        (1, 1, 0): 6,
        (1, 1, 1): 7,
    }

    expected = [(0, 4), (1, 5), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7)]

    seen = [(n_to_index[u], n_to_index[v]) for u, v in tree]
    seen = [(u, v) if u < v else (v, u) for u, v in seen]

    expected = sorted(expected, key=lambda x: (x[0], x[1]))
    seen = sorted(seen, key=lambda x: (x[0], x[1]))

    assert expected == seen


def test_maximin_tree_0_connector():
    intensities = np.array([[[1.0, 3.0, 2.0], [7.0, 0.0, 6.0]]], dtype=float,)

    tree = maximin_tree_edges(intensities)

    n_to_index = {
        (0, 0, 0): 0,
        (0, 0, 1): 1,
        (0, 0, 2): 2,
        (0, 1, 0): 3,
        (0, 1, 1): 4,
        (0, 1, 2): 5,
    }

    expected = [
        (2, 5),  # cost (-2, -6)
        (1, 2),  # cost (-2, -3)
        (0, 3),  # cost (-1, -7)
        (0, 1),  # cost (-1, -3)
        (3, 4),  # cost (-0, -7)
        ]

    seen = [(n_to_index[u], n_to_index[v]) for u, v in tree]
    seen = [(u, v) if u < v else (v, u) for u, v in seen]

    assert expected == seen
