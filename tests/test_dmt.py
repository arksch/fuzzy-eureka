"""
Tests
"""
import numpy as np
import pytest


parametrize = pytest.mark.parametrize


def test_morse_reduce_circle():
    """ Refactored version is like [MN13] """
    from dmt.dmt import morse_reduce
    from dmt.morse_reduce_nanda import morse_reduce_nanda
    from dmt.morse_complex import MorseComplex
    # Circle
    morse_complex = MorseComplex([[0, 0, 0, 1, 0, 1],
                                  [0, 0, 0, 1, 1, 0],
                                  [0, 0, 0, 0, 1, 1],
                                  [0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0]],
                                 cell_dimensions=[0, 0, 0, 1, 1, 1])
    reduced_cplx, matching = morse_reduce_nanda(morse_complex, return_matching=True)
    assert np.all(reduced_cplx.boundary_matrix == np.array([[0, 0],
                                                            [0, 0]]))
    assert np.all(reduced_cplx.cell_dimensions == [0, 1])
    # Testing whether our custom reduction gets the same result as [MN13]
    reduced_cplx2, matching2 = morse_reduce(morse_complex, return_matching=True)
    assert matching.matches == matching2.matches
    assert np.all(reduced_cplx.boundary_matrix == reduced_cplx2.boundary_matrix)
    assert np.all(reduced_cplx.cell_dimensions == reduced_cplx2.cell_dimensions)


def test_morse_reduce_two_triangles():
    """ Refactored version is like [MN13] """
    from dmt.dmt import morse_reduce
    from dmt.morse_reduce_nanda import morse_reduce_nanda
    from dmt.morse_complex import MorseComplex
    # Disc from two triangles      a  b  c  d  ab bc ac bd cd bdc abc
    morse_complex = MorseComplex([[0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],  # a
                                  [0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0],  # b
                                  [0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0],  # c
                                  [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],  # d
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # ab
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],  # bc
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # ac
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # bd
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # cd
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # bdc
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],  # abc
                                 cell_dimensions=[0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2])
    reduced_cplx, matching = morse_reduce_nanda(morse_complex, return_matching=True)
    assert np.all(reduced_cplx.boundary_matrix == np.array([[0]]))
    assert np.all(reduced_cplx.cell_dimensions == [0])
    # Testing whether our custom reduction gets the same result as [MN13]
    reduced_cplx2, matching2 = morse_reduce(morse_complex, return_matching=True)
    assert matching.matches == matching2.matches
    assert np.all(reduced_cplx.boundary_matrix == reduced_cplx2.boundary_matrix)
    assert np.all(reduced_cplx.cell_dimensions == reduced_cplx2.cell_dimensions)


def test_filtered_morse_reduce():
    """ Curiously stupid performance of the Perseus algorithm for a VR-cplx from a square

    Only the tetrahedron is reduced with a triangle.
    No triangle is reduced with an edge, as the triangles appear at filtration value sqrt(2),
    together with the two diagonals. But no triangle is bounded by both diagonals. In the beginning a
    diagonal is picked as an ace and the algorithm will not find a triangle that bounds the other diagonal.
    The only possible reduction is the tetrahedron with one of the triangles. All the other simplices
    (the other diagonal and three triangles) will be marked as aces.
    """
    from dmt.dmt import construct_acyclic_matching, reduce_by_acyclic_matching
    from dmt.morse_reduce_nanda import morse_reduce_nanda
    from dmt.cechmate_wrap import VietorisRips
    from dmt.pers_hom import filter_zero_persistence_classes
    points = [[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]]
    cplx = VietorisRips(points, max_dimension=3)
    reduced_cplx, matching = morse_reduce_nanda(cplx, return_matching=True)
    assert len(matching.matches) == 1
    assert reduced_cplx.size < cplx.size

    matching_unfiltered = construct_acyclic_matching(cplx)
    matching_filtered = matching_unfiltered.induce_filtered_matching(cplx.filtration)
    assert len(matching_filtered.matches) == 2
    reduced_cplx2 = reduce_by_acyclic_matching(matching_filtered)
    assert reduced_cplx2.size < cplx.size

    pers_diag1 = reduced_cplx.persistence_diagram()
    pers_diag2 = reduced_cplx2.persistence_diagram()
    # Results differ only by a 1-cycle of zero persistence, as expected
    assert len(pers_diag1[1]) == len(pers_diag2[1]) + 1
    for dim in [0, 1, 2]:
        assert np.all(filter_zero_persistence_classes(pers_diag1)[dim] == filter_zero_persistence_classes(pers_diag2)[dim])


def test_real_life_morse_reduce():
    from dmt import MorseComplex
    from dmt.data import load_complex
    from dmt.dmt import morse_reduce
    from dmt.binning import get_binning
    from dmt.morse_reduce_nanda import morse_reduce_nanda
    # This once broke
    delta = 0.05
    cplx = load_complex("normal_dist_2D_100pts_1.csv")
    binned_filtration = get_binning(cplx.filtration, delta)
    cplx_discrete = MorseComplex(cplx.boundary_matrix_csr, cplx.cell_dimensions,
                                 filtration=binned_filtration, points=cplx.points)
    cplx_nanda, matching_nanda = morse_reduce_nanda(cplx_discrete, return_matching=True)
    cplx_we, matching_we = morse_reduce(cplx_discrete, return_matching=True)
    assert cplx_nanda.valid_boundary()
    assert cplx_we.valid_boundary()


@parametrize("reduction_algo",
             [
                 ("sparse"),
                 ("dense")
             ])
def test_reduce_by_acyclic_matching_gradient_path_multiplicity(reduction_algo):
    """ Minimal example where there is a gradient path multiplicity of two

    Four triangles like this that are collapsed from left to right, with only the leftmost cell remaining
      ______
     / | >  \
    .  >  |––|
     \_|_>__/
    """
    from dmt import MorseComplex
    from dmt.matching import Matching
    from dmt.dmt import reduce_by_acyclic_matching, reduce_by_acyclic_matching_dense
    from dmt.pers_hom import filter_zero_persistence_classes
    algo_map = {"sparse": reduce_by_acyclic_matching,
                "dense": reduce_by_acyclic_matching_dense}
    edges = np.array([[1, 1, 0, 1, 0, 0, 1, 0],
                      [1, 0, 1, 0, 0, 1, 0, 1],
                      [0, 1, 1, 0, 1, 0, 0, 0],
                      [0, 0, 0, 1, 1, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 1]])
    triangles = np.array([[1, 1, 0, 0],
                          [0, 1, 1, 0],
                          [0, 1, 0, 1],
                          [0, 0, 1, 0],
                          [0, 0, 1, 1],
                          [0, 0, 0, 1],
                          [1, 0, 0, 0],
                          [1, 0, 0, 0]])
    boundary = np.block([
        [np.zeros((edges.shape[0], edges.shape[0])), edges, np.zeros((edges.shape[0], triangles.shape[1]))],
        [np.zeros((triangles.shape[0], edges.shape[0])), np.zeros((triangles.shape[0], edges.shape[1])), triangles],
        [np.zeros((triangles.shape[1], 17))]])
    cell_dimensions = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2]
    filtration =      [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1]
    cplx = MorseComplex(boundary.astype(bool), cell_dimensions, filtration=filtration, sort_by_filtration=False)
    matching = Matching(cplx, matches=[(5, 14), (6, 15), (7, 16)])
    """
matrix([[0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    """
    assert cplx.valid_boundary()

    reduced_cplx = algo_map[reduction_algo](matching)
    assert reduced_cplx.size < cplx.size
    assert reduced_cplx.valid_boundary()
    dgm = filter_zero_persistence_classes(cplx.persistence_diagram())
    reduced_dgm = filter_zero_persistence_classes(reduced_cplx.persistence_diagram())
    for i in [0, 1]:
        assert np.all(dgm[i] == reduced_dgm[i])


@parametrize("samples, dim, delta",
             [(10, 2, 0.0),
              (10, 2, 0.1),
              (10, 2, 0.5)])
def test_construct_acyclic_matching_with_approximate_filtration(samples, dim, delta):
    import numpy as np
    from dmt.dmt import construct_acyclic_matching_along_gradients
    from dmt.cechmate_wrap import AlphaComplex
    cplx = AlphaComplex(np.random.randn(samples, dim))
    original_filtration = cplx.filtration.copy()
    ac_m = construct_acyclic_matching_along_gradients(cplx, delta)
    approx_filtration = ac_m.morse_complex.filtration
    assert np.all(cplx.filtration == original_filtration)
    if delta > 0:
        assert np.any(approx_filtration != original_filtration)
    assert max(abs(approx_filtration - original_filtration)) <= delta, "Dont change filtration by more than delta"
    assert all([approx_filtration[q] - approx_filtration[k] == 0 for q, k in ac_m]), "Gradient paths are along constant new filtration values"
    assert cplx.valid_filtration(approx_filtration)


def test_morse_reduce_approx_does_not_change_filtration():
    import numpy as np
    from dmt.dmt import morse_approx_along_gradients
    from dmt.cechmate_wrap import AlphaComplex
    cplx = AlphaComplex(np.random.randn(10, 2))
    original_filtration = cplx.filtration.copy()
    reduced_cplx = morse_approx_along_gradients(cplx, delta=.5)
    assert np.all(cplx.filtration == original_filtration)


def test_acyclic_matching_with_delta():
    from dmt.dmt import construct_acyclic_matching
    from dmt.cechmate_wrap import VietorisRips
    points = [[0, 0],
              [0, 1],
              [2, 0]]
    cplx = VietorisRips(points, max_dimension=3)
    # >>> cplx.boundary_matrix.astype(int)
    # array([[0, 0, 0, 1, 1, 0, 0],
    #        [0, 0, 0, 1, 0, 1, 0],
    #        [0, 0, 0, 0, 1, 1, 0],
    #        [0, 0, 0, 0, 0, 0, 1],
    #        [0, 0, 0, 0, 0, 0, 1],
    #        [0, 0, 0, 0, 0, 0, 1],
    #        [0, 0, 0, 0, 0, 0, 0]])
    # >>> cplx.filtration
    # array([0., 0., 0., 1., 2., 2.23606798, 2.23606798])
    assert construct_acyclic_matching(cplx).matches == [(1, 3), (2, 4), (5, 6)]
    assert construct_acyclic_matching(cplx, delta=1.5).matches == [(1, 3), (5, 6)]
    assert construct_acyclic_matching(cplx, delta=.5).matches == [(5, 6)]
    assert construct_acyclic_matching(cplx, delta=0).matches == [(5, 6)]
