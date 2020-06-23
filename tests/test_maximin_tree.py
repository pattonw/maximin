import numpy as np

from maximin import maximin_tree_query


def test_maximin_query():
    intensities = np.array(
        [[[7.0, 6.0], [5.0, 4.0]], [[0.0, 1.0], [3.0, 2.0]], [[7.0, 6.0], [5.0, 4.0]]],
        dtype=float,
    )
    #                  0  1    2  3      4  5    6  7      8  9   10 11
    mask = np.array([[[0, 1], [0, 0]], [[0, 0], [1, 0]], [[1, 0], [0, 1]]], dtype=np.uint8)

    costs = maximin_tree_query(intensities, mask, True)

    n_to_index = {
        (0, 0, 0): 0,
        (0, 0, 1): 1,
        (0, 1, 0): 2,
        (0, 1, 1): 3,
        (1, 0, 0): 4,
        (1, 0, 1): 5,
        (1, 1, 0): 6,
        (1, 1, 1): 7,
        (2, 0, 0): 8,
        (2, 0, 1): 9,
        (2, 1, 0): 10,
        (2, 1, 1): 11,
    }

    expected_costs = {
        (1, 6): 3,
        (1, 8): 3,
        (1, 11): 3,
        (6, 8): 3,
        (6, 11): 3,
        (8, 11): 4,
    }

    seen = [(n_to_index[u], n_to_index[v], cost) for u, v, cost in costs]
    seen = [(u, v, c) if u < v else (v, u, c) for u, v, c in seen]
    seen = {(u, v): c for u, v, c in seen}

    assert len(seen) == 3

    for key, value in seen.items():
        assert expected_costs[key] == value

