"""
Basic classes and methods for persistent homology
Not optimized for speed, but good for testing
"""
import warnings

import numpy as np
from numpy.linalg import matrix_rank
try:
    from gudhi import bottleneck_distance as bottleneck_external
except ImportError:
    from persim import bottleneck as bottleneck_external
    warnings.warn("Could not import gudhi.bottleneck_distance. Is Gudhi installed?\n"
                  "Using persim.bottleneck instead, which returns bad results at times.")


class DenseComplex(object):
    """ Implements a complex with dense boundary matrices, separated by dimension

    This is useful to have an easy computation of persistent homology for testing
    """

    def __init__(self, boundary_matrices):
        """ Constructor

        :param boundary_matrices: Boundary matrices by dimension
        """
        self.boundary_matrices = list(map(np.array, boundary_matrices))
        if self.boundary_matrices[0].shape[0]:
            # Non zero boundary, so probably need to add the zeroeth boundary matrix
            self.boundary_matrices = [np.empty((0, self.boundary_matrices[0].shape[0]))] + self.boundary_matrices
        self.max_dimension = len(self.boundary_matrices) - 1
        # Index -1 smartly returns 0 for empty arrays and [1] for others without IndexError
        self.dim_cell_counts = [b.shape[-1] for b in self.boundary_matrices]

    @property
    def well_defined(self):
        """ Checks if the square of the boundary is zero """
        return np.all([np.all(a @ b == 0) for (a, b)
                       in zip(self.boundary_matrices[:-1], self.boundary_matrices[1:])])

    def boundary_rank(self, dim):
        """ Rank of the boundary of the given dimension

        :param dim: Dimension of the boundary operator """
        if (dim <= 0) | (self.max_dimension < dim):
            return 0
        return matrix_rank(self.boundary_matrices[dim])

    def betty(self, dim):
        """ Betty number of the given dimension """
        if (dim < 0) | (self.max_dimension < dim):
            return 0
        return self.dim_cell_counts[dim] - self.boundary_rank(dim) - self.boundary_rank(dim + 1)


class FilteredDenseComplex(DenseComplex):

    def __init__(self, filtration, boundary_matrices):
        """ Constructor

        :param filtration: A list of numbers the same length as the complex
        :param boundary_matrices:
        """
        super().__init__(boundary_matrices)
        self.filtration = filtration

    @property
    def well_defined(self):
        """ Checks if square of the boundary is zero and the filtration has the correct length """
        return super().well_defined & (sum(self.dim_cell_counts) == len(self.filtration))


def low_one(array1d):
    """ Gets lowest one in a 1d array """
    non_zero_indices = np.where(array1d != 0)[0]
    if not len(non_zero_indices):
        return -1
    return non_zero_indices[-1]


def left_right_reduction(cplx):
    """ Implements the standard algorithm from Edelsbrunner & Harer, no optimization
    :param cplx:
    :return:
    """
    # Initialize low ones for dimension zero
    all_low_ones = -np.ones(cplx.dim_cell_counts[0])
    simplices_dones = 0
    for dim in range(1, cplx.max_dimension + 1):
        low_ones = np.nan * np.empty(cplx.dim_cell_counts[dim])
        boundary_matrix = cplx.boundary_matrices[dim]
        for right_simplex in range(cplx.dim_cell_counts[dim]):
            low_ones[right_simplex] = low_one(boundary_matrix[:, right_simplex])
            left_simplex = 0
            while left_simplex < right_simplex and low_ones[right_simplex] != -1:
                if low_ones[right_simplex] == low_ones[left_simplex]:
                    # bitwise_xor is Z_2 subtraction
                    boundary_matrix[:, right_simplex] = np.bitwise_xor(boundary_matrix[:, right_simplex],
                                                                       boundary_matrix[:, left_simplex])
                    low_ones[right_simplex] = low_one(boundary_matrix[:, right_simplex])
                    left_simplex = 0
                else:
                    left_simplex += 1
        cplx.boundary_matrices[dim] = boundary_matrix
        all_low_ones = np.append(all_low_ones, low_ones + simplices_dones)
        simplices_dones += cplx.dim_cell_counts[dim - 1]
    return all_low_ones.astype(int)


def persistence_diagram(filtered_complex):
    """ Computes the persistence diagram

    Assumes a properly sorted filtered complex, dimension first, filtration second"""
    low_ones = left_right_reduction(filtered_complex)
    cell_dimensions = sum([[dim] * filtered_complex.dim_cell_counts[dim]
                           for dim in range(filtered_complex.max_dimension + 1)], [])
    pers_diag = {dim: [] for dim in range(filtered_complex.max_dimension + 1)}
    killed_cells = set(low_ones)  # Also includes -1, but who cares
    for cell_ix in range(len(low_ones)):
        if cell_ix not in killed_cells:
            cell_dim = cell_dimensions[cell_ix]
            cell_filtration = filtered_complex.filtration[cell_ix]
            if low_ones[cell_ix] == -1:
                # Lives until infinity
                cycle_dim = cell_dim
                birth = cell_filtration
                death = np.inf
            else:
                cycle_dim = cell_dim - 1
                birth = filtered_complex.filtration[low_ones[cell_ix]]
                death = cell_filtration
            pers_diag[cycle_dim].append([birth, death])
    for dim in pers_diag:
        pers_diag[dim] = np.array(pers_diag[dim])
    return pers_diag


def filter_zero_persistence_classes(pers_diag):
    """ Helper to remove zero persistence classes """
    for dim in pers_diag:
        if len(pers_diag[dim]):
            pers_diag[dim] = pers_diag[dim][pers_diag[dim][:, 0] != pers_diag[dim][:, 1]]
    return pers_diag


def bottleneck(dgm1, dgm2, degree=None):
    """ Bottleneck distance

    Wraps gudhi's bottleneck

    :param dgm1: Persistence diagram
    :param dgm2: Persistence diagram
    :param degree: Return bottleneck distance of a specific degree only,
    otherwise returns maximal bottleneck distance
    :returns: Bottleneck distance
    """
    # return the distance to the empty diagram
    def empty_bottleneck(dgm1, dgm2):
        empty_dgm = np.array([[0, 0]])
        return bottleneck_external(dgm1 if dgm1.size else empty_dgm, dgm2 if dgm2.size else empty_dgm)
    if degree:
        dims = [degree]
    else:
        dims = range(max(list(dgm1.keys()) + list(dgm2.keys())))
    empty_a = np.empty(0)
    return max([empty_bottleneck(dgm1.get(i, empty_a), dgm2.get(i, empty_a)) for i in dims])
