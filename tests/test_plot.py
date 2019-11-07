""" Tests """
import pytest


parametrize = pytest.mark.parametrize


@parametrize("lines, expected",
             [
                 ([[1, 2], [2, 3], [3, 1]], [1, 2, 3]),
                 ([[1, 3], [2, 3], [1, 2]], [1, 3, 2]),
                 ([[], [1, 2], [2, 3], [3, 1]], [1, 2, 3]),
                 ([[30, 48], [ 9, 36], [35, 49], [], [13, 36], [34, 48], [27, 30], [29, 49], [13, 34],
                   [28, 51], [29, 51], [ 9, 28], [27, 35]],
                  [30, 48, 34, 13, 36, 9, 28, 51, 29, 49, 35, 27])
             ])
def test_sorted_boundary(lines, expected):
    import numpy as np
    from dmt.plot import sorted_boundary
    lines = list(map(np.array, lines))
    result = sorted_boundary(lines)
    assert result == expected
