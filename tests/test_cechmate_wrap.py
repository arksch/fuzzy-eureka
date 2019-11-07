"""
Tests
"""
from tempfile import mkstemp

import numpy as np


def test_cech_complex():
    from dmt.cechmate_wrap import CechComplex
    points = [[0, 0],
              [1, 2],
              [2, 0]]
    cplx = CechComplex(points)
    assert cplx.boundary_matrix.shape == (7, 7)
    # This matrix is sorted by dimension and filtration value
    assert np.all(cplx.boundary_matrix.astype(int) ==
                  [[0, 0, 0, 1, 1, 0, 0],
                   [0, 0, 0, 0, 1, 1, 0],
                   [0, 0, 0, 1, 0, 1, 0],
                   [0, 0, 0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 0, 0, 0]])
    assert np.allclose(cplx.filtration, [0, 0, 0,
                                         1, .5 * 5 ** .5, .5 * 5 ** .5,
                                         1.25])


def test_parse_cechmate():
    import numpy as np
    from dmt.cechmate_wrap import parse_cechmate
    from dmt.morse_complex import MorseComplex
    cechmate = [([0], 0),  # 0
                 ([1], 0),  # 1
                 ([2], 0),  # 2
                 ([3], 0),  # 3
                 ((0, 1, 3), 3.5347319838680913),  # 12
                 ((1, 3), 3.5347319838680913),  # 11
                 ((0, 1, 2), 1.760962625882297),  # 10
                 ((1, 2), 1.760962625882297),  # 9
                 ((0, 2, 3), 1.0504164256818944),  # 8
                 ((0, 3), 1.0504164256818944),  # 7
                 ((0, 1), 0.2489387964292784),  # 4
                 ((0, 2), 0.30122587679897417),  # 5
                 ((2, 3), 0.6116032053615759)]  #  6
    cplx = parse_cechmate(cechmate)

    # Hacky way to compute the bitwise_xor matrix multiplication
    bdry = cplx["boundary_matrix"]
    assert np.mod(bdry.astype(int).dot(bdry.astype(int)).todense(), 2).sum() == 0

    # Result is not yet sorted
    assert np.all(cplx["cell_dimensions"] == [0, 0, 0, 0, 2, 1, 2, 1, 2, 1, 1, 1, 1])
    assert np.all(cplx["filtration"] == [0, 0, 0, 0, 3.5347319838680913, 3.5347319838680913,
                                         1.760962625882297, 1.760962625882297, 1.0504164256818944,
                                         1.0504164256818944, 0.2489387964292784, 0.30122587679897417,
                                         0.6116032053615759])
    assert np.all(cplx["boundary_matrix"].todense().astype(int) ==
                  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                   [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1],
                   [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                   [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]])

    morsecplx = MorseComplex(**cplx)
    assert np.all(morsecplx.boundary_matrix==[[0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0],
                                              [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0],
                                              [0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0],
                                              [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0],
                                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                                              [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
                                              [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                                              [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])


def test_persistent_homology_cechmate():
    import numpy as np
    from cechmate import Alpha
    from dmt.cechmate_wrap import AlphaComplex
    from dmt.pers_hom import filter_zero_persistence_classes
    points = np.random.randn(10, 2)
    cplx = AlphaComplex(points)
    cechmate_dgms = Alpha().diagrams(cplx.cechmate_complex)
    my_dgms = filter_zero_persistence_classes(cplx.persistence_diagram())
    my_dgms[0] = my_dgms[0][my_dgms[0][:, 1] < np.inf]  # Cechmate computes reduced persistence
    for cechmate_dgm, my_dgm in zip(cechmate_dgms, my_dgms.values()):
        sort_this = lambda a: a[np.lexsort((a[:, 0], a[:, 1]))]
        assert np.allclose(sort_this(cechmate_dgm), sort_this(my_dgm))


def test_construct_vietoris_rips():
    """ Constructing a VR complex from these points, having the following distances
    1: 01, 02
    sqrt(2): 12, 23
    2: 13
    sqrt(5): 03

    1       3

    0   2

    """
    from dmt.cechmate_wrap import VietorisRips
    points = [[0, 0],
              [0, 1],
              [1, 0],
              [2, 1]]
    cplx = VietorisRips(points)
    cplx = cplx.sort_by_dimension()
    assert cplx.boundary_matrix.shape == (15, 15)
    # This matrix is sorted by dimension and filtration value
    assert np.all(cplx.boundary_matrix.astype(int) ==
                  [[0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    assert np.all(cplx.filtration == [0, 0, 0, 0,
                                      1, 1, 2**.5, 2**.5, 2, 5**.5,
                                      2**.5, 2, 5**.5, 5**.5,
                                      5**.5])
    assert np.all(cplx.get_boundary(14) == [10, 11, 12, 13])
    assert np.all(cplx.get_coboundary(2) == [5, 6, 7])


def test_vietoris_rips_io_brips():
    import os
    from dmt.cechmate_wrap import VietorisRips
    points = [[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]]
    cplx = VietorisRips(points, max_dimension=3)
    tmp_file, tmp_filename = mkstemp()
    cplx.save_brips(tmp_filename)
    cplx_roundway = VietorisRips.load_brips(tmp_filename, max_dimension=3)
    assert np.all(cplx.boundary_matrix == cplx_roundway.boundary_matrix)
    assert np.all(cplx.filtration == cplx_roundway.filtration)
    assert np.all(cplx.cell_dimensions == cplx_roundway.cell_dimensions)
    os.close(tmp_file)
    os.remove(tmp_filename)
