"""
Tests
"""


def test_perseus_unreduced_count():
    import os
    from tempfile import mkstemp
    from dmt.perseus import to_nmfsimtop, perseus_unreduced_count
    cechmate_complex = [([0], 0), ([1], 0), ([2], 0), ((0, 1), 0), ((0, 2), 1)]
    tmp_file, tmp_filename = mkstemp()
    to_nmfsimtop(cechmate_complex, tmp_filename)
    count = perseus_unreduced_count(tmp_filename)
    os.close(tmp_file)
    os.remove(tmp_filename)
    assert count == 3


def test_perseus_removes_files():
    import os
    from tempfile import mkstemp
    from dmt.perseus import to_nmfsimtop, perseus_unreduced_count
    current_dir = os.listdir(os.getcwd())
    cechmate_complex = [([0], 0), ([1], 0), ([2], 0), ((0, 1), 0), ((0, 2), 1)]
    tmp_file, tmp_filename = mkstemp()
    to_nmfsimtop(cechmate_complex, tmp_filename)
    count = perseus_unreduced_count(tmp_filename)
    os.close(tmp_file)
    os.remove(tmp_filename)
    assert sorted(current_dir) == sorted(os.listdir(os.getcwd()))


def test_perseus_persistent_homology():
    import numpy as np
    from numpy.testing import assert_allclose
    from dmt.perseus import perseus_persistent_homology
    cechmate_complex = [([0], 0), ([1], 0), ([2], 0), ([3], 0),
                        ((0, 1), 1), ((1, 2), 2), ((2, 3), 3), ((0, 3), 4)]
    dgms = perseus_persistent_homology(cechmate_complex)
    assert_allclose(dgms[0], [[0, 1], [0, 2], [0, 3], [0, np.inf]])
    assert_allclose(dgms[1], [[4, np.inf]])


def test_perseus_persistent_homology_approx():
    import numpy as np
    from numpy.testing import assert_allclose
    from dmt.perseus import perseus_persistent_homology
    delta = 1
    cechmate_complex = [([0], 0), ([1], 0), ([2], 0), ([3], 0),
                        ((0, 1), 1), ((1, 2), 2), ((2, 3), 3), ((0, 3), 4)]
    dgms = perseus_persistent_homology(cechmate_complex, delta=delta, deltatrick=True)
    assert_allclose(dgms[0], [[1, 3], [1, 3], [1, np.inf]])
    assert_allclose(dgms[1], [[5, np.inf]])
