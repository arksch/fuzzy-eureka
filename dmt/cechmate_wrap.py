"""
Parser for cechmate format of simplicial complex
"""
from itertools import chain, combinations

from cechmate import Cech, Rips, Alpha
import numpy as np
from scipy.sparse import coo_matrix

from dmt.morse_complex import MorseComplex
from dmt.perseus import save_points_perseus_brips, load_points_perseus_brips


def parse_cechmate(cechmate_complex):
    """ Parses the Cechmate format for simplicial complexes

    :param cechmate_complex: [(simplex_as_index_tuple, filtration)]
    :return dict 'cell_dimensions': np.ndarray, 'filtration': np.ndarray,
     'boundary_matrix': scipy.sparse.coo_matrix, 'cechmate_complex': cechmate complex for testing

    :Example:
    >>> cechmate_cplx = [([0], 0), ([1], 0), ([2], 0), ((0, 1, 2), 1.760962625882297), ((1, 2), 1.760962625882297), ((0, 2), 0.30122587679897417), ((0, 1), 0.2489387964292784)]
    >>> MorseComplex(**parse_cechmate(cechmate_cplx)
    """
    simplices, filtration = zip(*cechmate_complex)
    simplices = list(map(tuple, simplices))  # All should be tuples, so they can be in a dict
    size = len(simplices)
    index_map = {splx: ix for splx, ix in zip(simplices, range(size))}
    columns_rows = chain.from_iterable([[(index_map[splx], index_map[bdry])
                                         for bdry in combinations(splx, len(splx) - 1) if bdry]
                                        for splx in simplices])
    columns, rows = zip(*columns_rows)
    columns, rows = list(columns), list(rows)
    data = [True] * len(columns)
    boundary = coo_matrix((data, (rows, columns)), shape=(size, size), dtype=bool)
    filtration = list(filtration)
    cell_dimensions = np.array(list(map(len, simplices))) - 1
    return dict(boundary_matrix=boundary,
                cell_dimensions=cell_dimensions,
                filtration=filtration,
                cechmate_complex=cechmate_complex)


class VietorisRips(MorseComplex):

    default_max_dim = 3

    def __init__(self, points, max_dimension=default_max_dim):
        points = np.array(points)
        self.max_dimension = max_dimension
        super().__init__(points=points, **parse_cechmate(Rips(maxdim=self.max_dimension).build(points)))

    def save_brips(self, filepath):
        save_points_perseus_brips(filepath, self.points)

    @classmethod
    def load_brips(cls, filepath, max_dimension=default_max_dim):
        return cls(load_points_perseus_brips(filepath), max_dimension)


class CechComplex(MorseComplex):

    default_max_dim = 3

    def __init__(self, points, max_dimension=default_max_dim):
        points = np.array(points, dtype=float)
        self.max_dimension = max_dimension
        super().__init__(points=points, **parse_cechmate(Cech(maxdim=self.max_dimension).build(points)))


class AlphaComplex(MorseComplex):

    def __init__(self, points):
        points = np.array(points, dtype=float)
        super().__init__(points=points, **parse_cechmate(Alpha().build(points)))
