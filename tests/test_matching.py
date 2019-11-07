"""
Tests
"""
import numpy as np


def test_get_ace_ixs():
    from dmt.morse_complex import MorseComplex
    from dmt.matching import Matching
    qk = np.random.choice(range(100), 10, replace=False)
    morse_complex = MorseComplex(boundary_matrix=np.empty((100, 100)),
                                 cell_dimensions=np.empty(100))
    matching = Matching(morse_complex=morse_complex,
                        matches=list(zip(qk[range(0, 10, 2)], qk[range(1, 10, 2)])))
    assert len(matching.matches) == 5
    assert len(matching.unmatched_ixs) == 90
