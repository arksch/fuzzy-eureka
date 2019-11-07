""" Wrapping PHAT """
import numpy as np
import phat


def persistence_diagram(boundary_matrix_csc, dimensions, filtration):
    """ Compute persistence diagram from a sparse matrix

    :param boundary_matrix_csc: Sparse matrix
    :param dimensions: Cell dimensions
    :param filtration: Filtration
    :returns: Persistence diagrams
    """
    sort_order = np.lexsort((dimensions, filtration))  # Last key has higher sort priority
    boundary_matrix_csc = boundary_matrix_csc[sort_order, :][:, sort_order]
    dimensions = dimensions[sort_order]
    filtration = filtration[sort_order]

    col_count = boundary_matrix_csc.shape[1]
    assert len(dimensions) == col_count
    columns = [boundary_matrix_csc.getcol(col).indices.tolist() for col in range(col_count)]
    # Many representations (vector_vector, full_pivot_column, bit_tree_pivot_column)
    # of PHAT seem to work incorrectly. vector_heap is ok.
    bdry = phat.boundary_matrix(representation=phat.representations.vector_heap,
                                columns=list(zip(dimensions, columns)))
    pairs = bdry.compute_persistence_pairs(reduction=phat.reductions.twist_reduction)
    dgms = pairs_to_diagram(pairs, dimensions, filtration)
    return dgms


def pairs_to_diagram(pairs, dimensions, filtration):
    """ Parse PHAT pairs into persistence diagram """
    pairs = np.array(pairs, dtype=int).reshape(-1, 2)  # Empty arrays have two columns
    pairs = add_infinite_pairs(pairs, size=len(filtration))
    return {dim: pairs_to_filtr_diag(pairs_in_dimension(pairs, dimensions, dim), filtration)
            for dim in range(max(dimensions) + 1)}


def add_infinite_pairs(pairs, size):
    """ Helper """
    infinite_births = np.setdiff1d(np.arange(size, dtype=int), pairs)
    infinite_pairs = np.column_stack([infinite_births, np.full(len(infinite_births), -1)])
    return np.concatenate([pairs, infinite_pairs])


def pairs_in_dimension(pairs, dimensions, dim):
    """ Helper """
    ixs = np.where(dimensions == dim)[0]
    return pairs[np.isin(pairs[:, 0], ixs)]


def pairs_to_filtr_diag(pairs, filtration):
    """ Helper """
    filtration_with_inf = np.append(filtration, np.inf)
    return filtration_with_inf[pairs.astype(int)]
