import numpy as np
import pytest

from maximin import maximin_tree


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

    tree = maximin_tree(intensities)

    expected_edges = ["ugh"]

    assert tree == expected_edges


def test_maximin_tree_2d():
    intensities = np.array([[0, 1], [3, 2]], dtype=float)

    with pytest.raises(Exception):
        tree = maximin_tree(intensities)


def test_maximin_tree_small():
    intensities = np.array([[[0, 1], [3, 2]]], dtype=float)

    tree = maximin_tree(intensities)

    expected_edges = [((0, 0, 0), (0, 1, 0)), ((0, 1, 0), (1, 1, 0)), ((1, 1, 0), (1, 0, 0))]

    assert expected_edges == tree



