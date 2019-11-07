"""
Data structure for MorseComplex
"""
import json

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, find

from dmt.filtration import Filtration, ApproxFiltration
from dmt.phat_wrap import persistence_diagram
from dmt.plot import complex_with_points_2D
from dmt.json_np import NumpyEncoder


class MorseComplex(object):
    """
    A very generic definition of a complex, inspired from [MN13]
    """

    def __init__(self, boundary_matrix=None, cell_dimensions=None, filtration=None,
                 cechmate_complex=None, points=None, sort_by_filtration=True):
        """ Data needed to create a morse complex

        :param boundary_matrix: Boundary matrix of the complex, can be sparse
        :param cell_dimensions: Dimensions of the cells
        :param filtration: Filtration values for the cells of a filtered morse complex
        :param cechmate_complex: Simplices and filtration values as given by cechmate.
        This is present only for simplicial complexes
        :param points: Underlying points corresponding to the zero cells
        :param sort_by_filtration: Sort the complex by filtration.
        Useful as this is assumed for many algorithms"""
        self._set_boundary(boundary_matrix)
        if cell_dimensions is None:
            cell_dimensions = np.empty(0)
        self.cell_dimensions = np.array(cell_dimensions)
        self.filtration = self._make_filtration(filtration)
        self.points = np.array(points) if points is not None else None
        self.cechmate_complex = cechmate_complex
        self._original_order = np.arange(self.size, dtype=int)
        self._original_order_to_points = np.argsort(self._dimension_order())
        if sort_by_filtration:
            self.sort_by_filtration()

    @property
    def size(self):
        return len(self.cell_dimensions)

    @property
    def complex_dimension(self):
        return max(self.cell_dimensions) if self.size else 0

    def _set_boundary(self, boundary_matrix):
        if boundary_matrix is None:
            boundary_matrix = np.empty((0, 0))
        self.boundary_matrix_csc = csc_matrix(boundary_matrix).astype(bool)  # Fast column access
        self.boundary_matrix_csr = csr_matrix(boundary_matrix).astype(bool)  # Fast row access

    def _make_filtration(self, filtration):
        if filtration is None:
            return Filtration(np.zeros(self.size))
        if type(filtration) not in [Filtration, ApproxFiltration]:
            return Filtration(filtration)
        return filtration

    def _dimension_order(self):
        return np.argsort(self.cell_dimensions, kind="stable")

    def _filtration_dimension_order(self):
        # np.lexsort sorting priority is from back to front (opposed to sorted())
        return np.lexsort((self.cell_dimensions, self.filtration))

    def _dimension_filtration_order(self):
        # np.lexsort sorting priority is from back to front (opposed to sorted())
        return np.lexsort((self.filtration, self.cell_dimensions))

    def copy(self, **kwargs):
        """ Return a copy of self

        :param kwargs: Keyword arguments to replace in the copy
        :returns: copy
        """
        copied = dict(boundary_matrix=self.boundary_matrix_csc,
                      cell_dimensions=self.cell_dimensions,
                      filtration=self.filtration,
                      cechmate_complex=self.cechmate_complex,
                      points=self.points,
                      sort_by_filtration=False)
        copied.update(kwargs)
        return MorseComplex(**copied)

    def get_boundary(self, cell_ix):
        """ Gets a cell's boundary

        :param cell_ix: Index of the cobounding cell
        :returns: Indices of the cells boundary
        """
        return self.boundary_matrix_csr[:, cell_ix].nonzero()[0]  # Why is this faster than csc according to cProfile?

    def get_coboundary(self, cell_ix):
        """ Gets a cell's coboundary

        :param cell_ix: Index of the bounding cell
        :returns: Indices of the cells coboundary
        """
        return self.boundary_matrix_csr[cell_ix, :].nonzero()[1]

    def get_point(self, cell_ixs):
        """ Gets the R^n point referred to by a zero cell """
        return self.points[self._original_order_to_points[self._original_order[cell_ixs]]]

    def __getitem__(self, indices):
        """ Allows *inplace* numpy style indexing, slicing and sorting

        :param indices: Indices to resort or slice
        :returns: Self"""
        self.cell_dimensions = self.cell_dimensions[indices]
        self.filtration = self.filtration[indices]
        self.boundary_matrix_csc = self.boundary_matrix_csc[indices, :][:, indices]
        self.boundary_matrix_csr = self.boundary_matrix_csr[indices, :][:, indices]
        self._original_order = self._original_order[indices]
        return self

    def sort_by_filtration(self):
        """ Sorts primary by filtration and secondary by dimension """
        return self.__getitem__(self._filtration_dimension_order())

    def sort_by_dimension(self):
        """ Sorts primary by filtration and secondary by dimension """
        return self.__getitem__(self._dimension_filtration_order())

    def valid_filtration(self, filtration=None):
        """ The filtration needs to respect the face order """
        if filtration is None:
            filtration = self.filtration
        return all([all([filtration[bdry_ix] <= filtration[cell_ix] for bdry_ix in self.get_boundary(cell_ix)])
                    for cell_ix in range(self.size)])

    def valid_boundary(self):
        """ Checks if the square of the boundary is zero """
        boundary_int_square = self.boundary_matrix_csc.astype(int).dot(self.boundary_matrix_csc.astype(int))
        values = find(boundary_int_square)[2]  # Returns triple Col_ixs, Row_ixs, Values
        return np.all(np.mod(values, 2) == 0)

    @property
    def boundary_matrix(self):
        """ Gets the dense boundary matrix """
        return self.boundary_matrix_csc.todense()

    def persistence_diagram(self):
        """ Computes the persistence diagram """
        return persistence_diagram(self.boundary_matrix_csc, self.cell_dimensions, self.filtration)

    def plot(self, **kwargs):
        """ See plot.complex_with_points2D """
        return complex_with_points_2D(self, **kwargs)

    def save(self, filepath):
        """ Saves the complex to a JSON file """
        with open(filepath, "w") as fout:
            json.dump({"boundary_matrix": self.boundary_matrix,
                       "cell_dimensions": self.cell_dimensions,
                       "filtration": self.filtration,
                       "cechmate_complex": self.cechmate_complex,
                       "points": self.points},
                      fout, cls=NumpyEncoder)

    @classmethod
    def load(cls, filepath):
        """ Loads the complex from a JSON file """
        with open(filepath, "r") as fin:
            data = json.load(fin)
        return MorseComplex(**data)
