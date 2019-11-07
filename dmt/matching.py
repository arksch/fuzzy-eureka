"""
Classes for filtered acyclic partial matchings
"""
import numpy as np

from dmt.morse_complex import MorseComplex


class Matching(object):
    """ A matching is a discrete vector field on a cell complex """

    def __init__(self, morse_complex=None, matches=None):
        self.morse_complex = MorseComplex() if morse_complex is None else morse_complex
        self.matches = matches or []

    def __iter__(self):
        return iter(self.matches)

    def append(self, match):
        self.matches.append(match)

    def induce_filtered_matching(self, filtration):
        """ Filter matches by a filtration

        Only matches that appear at the same filtration value survive.

        :param filtration: Filtration to be used for filtering the matches.
        """
        matches = [(q, k) for (q, k) in self.matches if filtration[q] == filtration[k]]
        return Matching(self.morse_complex.copy(filtration=filtration),
                        matches=matches)

    @property
    def unmatched_ixs(self):
        """ Find which cells are unmatched """
        return np.setdiff1d(range(self.morse_complex.size), sum(zip(*self), ()))
