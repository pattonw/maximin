import numpy as np

from maximin import maximin_tree_query_hd


def test_maximin_query():
    intensities = np.zeros((3, 3, 3, 3))
    for i in range(3):
        intensities[:, 0, 0, i] = np.array([0, 0, i ** 2])
        intensities[:, 0, 1, i] = np.array([0, i ** 2, 1])

    mask = np.zeros((3, 3, 3), dtype=np.uint8)
    mask[0, 0, :] = 1
    mask[0, 1, :] = 1

    costs = maximin_tree_query_hd(intensities, mask)

    n_to_index = {
        (0, 0, 0): 0,
        (0, 0, 1): 1,
        (0, 0, 2): 2,
        (0, 1, 0): 3,
        (0, 1, 1): 4,
        (0, 1, 2): 5,
    }

    expected_costs = {
        (0, 1): 1,
        (0, 2): 3,
        (0, 3): 1,
        (0, 4): 1,
        (0, 5): 3,
        (1, 2): 3,
        (1, 3): 0,
        (1, 4): 1,
        (1, 5): 3,
        (2, 3): 3,
        (2, 4): 3,
        (2, 5): 3,
        (3, 4): 1,
        (3, 5): 3,
        (4, 5): 3,
    }

    seen = [(n_to_index[u], n_to_index[v], cost) for u, v, cost in costs]
    seen = [(u, v, c) if u < v else (v, u, c) for u, v, c in seen]
    seen = {(u, v): c for u, v, c in seen}

    print(costs)

    assert len(seen) == 5

    for key, value in seen.items():
        assert expected_costs[key] == value, f"key: {key}, value: {value}"

