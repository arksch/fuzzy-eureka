"""
Tests
"""


def test_phat_persistence_diagram():
    import numpy as np
    from numpy.testing import assert_allclose
    from dmt.complexes import MorseComplex
    from dmt.phat_wrap import persistence_diagram
    # Filtered circle
    morse_complex = MorseComplex([[0, 0, 0, 1, 0, 1],
                                  [0, 0, 0, 1, 1, 0],
                                  [0, 0, 0, 0, 1, 1],
                                  [0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0]],
                                 cell_dimensions=[0, 0, 0, 1, 1, 1],
                                 filtration=[0, 0, 0, 1, 2, 3])
    dgms = persistence_diagram(morse_complex.boundary_matrix_csc, morse_complex.cell_dimensions,
                               morse_complex.filtration)
    assert_allclose(dgms[0], [[0, 1], [0, 2], [0, np.inf]])
    assert_allclose(dgms[1], [[3, np.inf]])


def test_phat():
    import numpy as np
    import phat
    columns = [[], [], [], [], [], [], [], [], [], [], [0, 7], [5, 9], [0, 2], [4, 8], [7, 8],
               [2, 9], [0, 9], [16, 12, 15], [6, 8], [6, 7], [14, 18, 19], [1, 6], [1, 4],
               [4, 6], [23, 18, 13], [7, 9], [25, 16, 10], [0, 8], [27, 14, 10],
               [23, 21, 22], [6, 9], [30, 25, 19], [5, 6], [30, 32, 11], [3, 5], [3, 6], [35, 32, 34], [2, 8], [37, 27, 12], [1, 3], [39, 21, 35], [2, 4], [41, 37, 13]]
    dimensions = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1,
       1, 1, 2, 1, 2, 1, 2, 2, 1, 2, 1, 2, 1, 1, 2, 1, 2, 1, 2, 1, 2]
    bdry = phat.boundary_matrix(representation=phat.representations.vector_heap,
                                columns=list(zip(dimensions, columns)))
    pairs = np.array(bdry.compute_persistence_pairs(reduction=phat.reductions.twist_reduction))
    assert np.all(pairs[:, 0] != 10), "First added edge should kill 7"