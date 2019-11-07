"""
Tests
"""
import numpy as np
import pytest


parametrize = pytest.mark.parametrize


def test_ceil():
    from dmt.binning import ceil
    filtration = np.array([5, 3, 7, 2, 43, 7, 3, 1, 5, -2])
    bins = np.array([0, 2, 5, 10])
    assert np.all(ceil(filtration, bins) == [5., 5., 10., 2., np.inf, 10., 5., 2., 5., 0.])


def test_ceil_equal():
    from dmt.binning import ceil
    filtration = np.array([5, 3, 7, 2, 43, 7, 3, 1, 5, -2])
    bins = np.unique(filtration)
    assert np.all(ceil(filtration, bins) == filtration)


@parametrize("filtration,delta", [
    ([0, 1, 1, 2, 1, 2], 1),
    ([0, 3, 3, 3], 1.2),
    ([0, 0, 0, 3], 1.2),
    ([2e-16, 0], 0.),   # Some filtrations might be off by a small margin due to Miniball
])
def test_find_discretization(filtration, delta):
    from dmt.filtration import Filtration
    from dmt.binning import find_discretization, ceil
    filtr_len = len(filtration)
    assert filtr_len % 2 == 0, "Filtration should give pairwise matches, thus has to have an even length"
    acyclic_matching = list(zip(range(0, filtr_len + 1, 2), range(1, filtr_len + 1, 2)))  # Match neighboring pairs
    filtration = Filtration(filtration)
    bins = find_discretization(acyclic_matching, filtration, delta)
    ceiled_filtration = ceil(filtration, bins)
    assert np.all(ceiled_filtration >= filtration)
    assert ceiled_filtration.delta <= delta


def test_find_discretization_is_good():
    from dmt.matching import Matching
    from dmt.binning import find_discretization
    from dmt.binning import ceil
    acyclic_matching = Matching(matches=[(0, 1), (2, 3), (4, 5)])
    filtration = np.array([0, 1, 1, 2, 1, 2])
    delta = 1.2
    bins = find_discretization(acyclic_matching, filtration, delta)
    # For a good binning of the given matching the following should hold
    assert len(bins) >= 2
    assert bins[0] <= 1
    assert bins[-1] >= 2
    assert len(acyclic_matching.induce_filtered_matching(ceil(filtration, bins)).matches) == 2


@parametrize("matches, filtration, delta, expected_matches",
             [
                 ([(4, 6), (11, 22), (10, 21), (14, 25), (15, 26), (18, 32), (13, 23), (16, 27), (19, 34),
                   (17, 31), (20, 40), (24, 41), (28, 42), (29, 44), (33, 46), (39, 55), (36, 48), (35, 49),
                   (37, 52), (30, 43), (38, 50)],
                  [0., 0., 0., 0., 0.,
                   0., 0.22220853, 0.57590976, 1.23114802, 1.51425671,
                   1.71353541, 1.80401254, 1.90530583, 2.08282587, 2.09598363,
                   3.05179802, 3.26434612, 3.27992591, 3.62102972, 3.80530396,
                   3.83218186, 1.71353541, 1.80401254, 2.08282587, 2.09598363,
                   2.09598363, 3.05179802, 3.26434612, 3.26434612, 3.27992591,
                   3.27992591, 3.27992591, 3.62102972, 3.62102972, 3.80530396,
                   3.80530396, 3.80530396, 3.83218186, 3.83218186, 3.83218186,
                   3.83218186, 2.09598363, 3.26434612, 3.27992591, 3.27992591,
                   3.27992591, 3.62102972, 3.80530396, 3.80530396, 3.80530396,
                   3.83218186, 3.83218186, 3.83218186, 3.83218186, 3.83218186,
                   3.83218186],
                  0.3, 21),
                 ([(0, 1)], [0.0, 0.22220853], 0.3, 1),
                 ([(0, 1)], [0.0, 0.22220853], 0, 0),
                 ([(0, 1), (2, 3)],
                  [0., .5, 1 - 1e-9, 1.5],
                  1., 2)  # Properly deal with open intervals and epsilon
              ])
def test_find_discretization_on_real_acyclic_matching(matches, filtration, delta, expected_matches):
    from dmt.matching import Matching
    from dmt.binning import find_discretization
    from dmt.binning import ceil
    matching = Matching(matches=matches)
    bins = find_discretization(matching, filtration, delta)
    ceiled_filtration = ceil(filtration, bins)
    assert np.all(ceiled_filtration >= filtration)
    assert ceiled_filtration.delta <= delta
    assert len(matching.induce_filtered_matching(ceil(filtration, bins)).matches) == expected_matches
