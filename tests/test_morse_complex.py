"""
Tests
"""
from tempfile import mkstemp

import pytest


parametrize = pytest.mark.parametrize


@parametrize("filtration, valid",
             [
                 ([0, 0, 1], True),
                 ([0, 1, 0], False),
             ])
def test_valid_filtration(filtration, valid):
    from dmt.morse_complex import MorseComplex
    boundary = [[0, 0, 1],
                [0, 0, 1],
                [0, 0, 0]]
    dims = [0, 0, 1]
    assert (MorseComplex(boundary, dims, filtration).valid_filtration() == valid)


def test_io():
    import os
    import numpy as np
    from dmt.cechmate_wrap import AlphaComplex
    cplx = AlphaComplex(np.random.randn(10, 2))
    tmp_file, tmp_filename = mkstemp()
    cplx.save(tmp_filename)
    cplx_roundway = AlphaComplex.load(tmp_filename)
    assert np.all(cplx.boundary_matrix == cplx_roundway.boundary_matrix)
    assert np.all(cplx.cell_dimensions == cplx_roundway.cell_dimensions)
    assert np.all(cplx.filtration == cplx_roundway.filtration)
    assert np.all(cplx.points == cplx_roundway.points)
    # This last line should also work, but cechmate has a horrible mixture of tuples and lists
    # assert cplx.cechmate_complex == cplx_roundway.cechmate_complex
    os.close(tmp_file)
    os.remove(tmp_filename)


@parametrize("boundary, cell_dimensions, filtration, expected_order, sort_by",
             [
                 ([[0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0]],
                  [0, 0, 0, 0, 1, 1],
                  [0, 0, .5, .5, 1, 1.5],
                  [0, 1, 2, 3],
                  "filtration"),

                 ([[0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0]],
                  [0, 0, 0, 0, 1, 1],
                  [0, .5, 0, .5, 1, 1.5],
                  [0, 2, 1, 3],
                  "filtration"),

                 ([[0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 0, 0]],
                  [0, 0, 0, 1, 0, 1],
                  [0, 1, 0, 1, .5, 1.5],
                  [0, 2, 3, 1],
                  "filtration"),

                 ([[0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 0, 0]],
                  [0, 0, 0, 1, 0, 1],
                  [0, 0, 0, 1, .5, 1.5],
                  [0, 1, 2, 3],
                  "dimension")
             ])
def test_point_order(boundary, cell_dimensions, filtration, expected_order, sort_by):
    import numpy as np
    from dmt.morse_complex import MorseComplex
    point_count = len([d for d in cell_dimensions if d == 0])
    points = (np.full((point_count, 2), 1).T * np.arange(point_count)).T
    # Constructor sorts by filtration
    cplx = MorseComplex(boundary, cell_dimensions, filtration=filtration, points=points)
    if sort_by == "dimension":
        cplx.sort_by_dimension()
    elif sort_by == "filtration":
        cplx = cplx.sort_by_filtration()
    assert np.all(cplx.get_point([i for i in range(cplx.size) if cplx.cell_dimensions[i] == 0]) ==
                  points[expected_order])
