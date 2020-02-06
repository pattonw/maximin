import numpy as np

from maximin.maximin import get_42, double


def test_get_42():
    assert get_42() == 42


def test_double():
    x = np.array([1, 2, 3], dtype=float)
    assert all(np.isclose(double(x), np.array([2, 4, 6])))

