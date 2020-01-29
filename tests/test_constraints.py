from funlib_match_helpers import rusty_121_t2s_node as f


def test_121_t2s_nodes():
    constraints = f(
        [10, 11, 12, 13, 14, 15, 16],
        [
            (1, 10),
            (1, 11),
            (1, None),
            (2, 10),
            (2, 11),
            (2, 12),
            (2, None),
            (3, 14),
            (3, None),
        ],
    )

    expected_constraints = [[0, 3], [1, 4], [5], [], [7], [], []]

    assert constraints == expected_constraints
