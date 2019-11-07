"""
Tests
"""
import numpy as np


def test_low_one():
    from dmt.pers_hom import low_one
    array1d = np.array([1, 0, 1, 0, 0, 0, 0])
    assert low_one(array1d) == 2
    array0 = np.array([0, 0])
    assert low_one(array0) == -1


def test_left_right_reduction():
    """ Reduction of a simplicial triangle """
    from dmt.pers_hom import DenseComplex, left_right_reduction
    edges = [[1, 1, 0],
             [1, 0, 1],
             [0, 1, 1]]
    triangle = [[1],
                [1],
                [1]]
    cplx = DenseComplex([edges, triangle])
    low_ones = left_right_reduction(cplx)
    assert np.all(low_ones == [-1, -1, -1, 1, 2, -1, 5])
    assert np.all(cplx.boundary_matrices[1] == [[1, 1, 0],
                                                [1, 0, 0],
                                                [0, 1, 0]])
    assert np.all(cplx.boundary_matrices[2] == triangle)


def test_persistence_empty():
    """ Reduction of a simplicial triangle """
    from dmt.pers_hom import FilteredDenseComplex, persistence_diagram
    edges = []
    filtration = []
    filtered_complex = FilteredDenseComplex(filtration=filtration, boundary_matrices=[edges])
    pers_diag = persistence_diagram(filtered_complex)
    assert np.allclose(pers_diag[0], [])


def test_persistence_diagram_triangle():
    """ Reduction of a simplicial triangle """
    from dmt.pers_hom import FilteredDenseComplex, persistence_diagram
    edges = [[1, 1, 0],
             [1, 0, 1],
             [0, 1, 1]]
    triangle = [[1],
                [1],
                [1]]
    filtration = [0, 0, 0, 1, 2, 3, 4]
    filtered_complex = FilteredDenseComplex(filtration=filtration, boundary_matrices=[edges, triangle])
    pers_diag = persistence_diagram(filtered_complex)
    assert np.allclose(pers_diag[0], [[0, np.inf], [0, 1], [0, 2]])
    assert np.allclose(pers_diag[1], [[3, 4]])
    assert np.allclose(pers_diag[2], [[]])


def test_betty_circle():
    """
    Complex of a circle

    X----
    |    |
    -----X

    """
    from dmt.pers_hom import DenseComplex
    D1 = [[1, 1], [1, 1]]
    circle_cplx = DenseComplex([D1])
    assert circle_cplx.well_defined
    assert circle_cplx.betty(0) == 1
    assert circle_cplx.betty(1) == 1
    assert circle_cplx.betty(2) == 0


def test_betty_S2():
    """
    Compute S^2 betti numbers of some cell decomposition
    """
    from dmt.pers_hom import DenseComplex
    D1 = [[0]]
    D2 = [[1, 1]]
    sphere_cplx = DenseComplex([D1, D2])
    assert sphere_cplx.well_defined
    assert sphere_cplx.betty(0) == 1
    assert sphere_cplx.betty(1) == 0
    assert sphere_cplx.betty(2) == 1
    assert sphere_cplx.betty(3) == 0


def test_bottleneck():
    dgm = {0: np.array([[0., np.inf],
               [0., 0.10649897],
               [0., 0.11169725],
               [0., 0.11399432],
               [0., 0.11822441],
               [0., 0.12739568],
               [0., 0.12817322],
               [0., 0.12989155],
               [0., 0.13576328],
               [0., 0.15676248],
               [0., 0.15702479],
               [0., 0.15717725],
               [0., 0.16699927],
               [0., 0.18646686],
               [0., 0.18701381],
               [0., 0.18703563],
               [0., 0.18815514],
               [0., 0.20160782],
               [0., 0.2016301],
               [0., 0.20271505],
               [0., 0.21349704],
               [0., 0.21352366],
               [0., 0.21485717],
               [0., 0.21744589],
               [0., 0.22396986],
               [0., 0.22467735],
               [0., 0.23257689],
               [0., 0.23366344],
               [0., 0.2339978],
               [0., 0.24690084],
               [0., 0.25045701],
               [0., 0.2526927],
               [0., 0.28148942],
               [0., 0.30105915],
               [0., 0.36016472],
               [0., 0.38585473],
               [0., 0.45075592],
               [0., 0.46517901],
               [0., 0.48548894]]),
           1: np.array([[0.33859722, 0.41678413],
                        [0.30734239, 0.45291725],
                        [0.41773718, 0.47120808]]),
           2: np.array([])}
    from dmt.pers_hom import bottleneck
    assert 0 <= bottleneck(dgm, dgm) < 1e-9
