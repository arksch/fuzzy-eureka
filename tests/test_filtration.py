"""
Tests
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_filtration():
    from dmt.filtration import Filtration
    filtration = Filtration([1, 2, 3])
    assert_allclose(filtration[[0, 2]], [1, 3])


def test_approx_filtration():
    from dmt.filtration import ApproxFiltration
    approx_filtr = ApproxFiltration([1, 2, 3], [.9, 2.2, 3.5])
    assert approx_filtr.delta == .5


def test_approx_filtration_exact_type():
    from dmt.filtration import ApproxFiltration, Filtration
    approx_filtr = ApproxFiltration([1, 2, 3], Filtration([.9, 2.2, 3.5]))
    assert approx_filtr.delta == .5


def test_approx_filtration_slicing():
    from dmt.filtration import ApproxFiltration
    approx_filtr = ApproxFiltration([1, 2, 3], [.9, 2.2, 3.5])
    assert_allclose(approx_filtr[0:2].delta, .2)


def test_filtration_hashable_type():
    from dmt.filtration import Filtration
    filtration = Filtration([1, 2, 3, 3])
    assert np.all(np.unique(filtration) == [1, 2, 3])


@pytest.mark.parametrize("approx, exact, expected",
                         [
                             ([1, 2, 3, 3], [2, 3, 4, 4], [1, 2, 3]),
                             ([1, 2, 3, 3], None, [1, 2, 3])
                         ])
def test_approx_filtration_hashable_type(approx, exact, expected):
    from dmt.filtration import ApproxFiltration
    filtration = ApproxFiltration(approx, exact=exact)
    assert np.all(np.unique(filtration) == expected)
