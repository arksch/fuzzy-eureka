"""
Discrete Morse Theory and Approximation
"""
import numpy as np

from dmt.filtration import ApproxFiltration
from dmt.binning import get_binning, ceil, find_discretization
from dmt.matching import Matching


def morse_approx_binning(morse_complex, delta, return_matching=False, return_binned_complex=False,
                         deltatrick=True, epsilon=1e-9):
    """ Creates an approximative Morse reduction with the binning approach

    First constructs a regular delta-binning, discretizes the filtration along it,
    then reduces with the algorithm proposed by [MN13]

    :param morse_complex: MorseComplex to reduce
    :param delta: Approximation parameter for the filtration and the persistence module
    :param return_matching: Return a tuple of the reduced complex and the matching used to reduce it
    :param return_binned_complex: Return the binned complex as well
    :param deltatrick: Use 2*delta approximation from below and shift it down by delta. This allows more reductions.
    :returns Reduced complex, optionally also the filtered acyclic matching
    """
    binned_filtration = get_binning(morse_complex.filtration, delta,
                                    deltatrick=deltatrick, epsilon=epsilon)
    binned_cplx = morse_complex.copy(filtration=binned_filtration)
    if return_binned_complex:
        return tuple(morse_reduce(binned_cplx, return_matching=return_matching)) + (binned_cplx,)
    return morse_reduce(binned_cplx, return_matching=return_matching)


def morse_approx_induced_matching(morse_complex, delta, return_matching=False, deltatrick=True):
    """ Creates an approximative Morse reduction with the induced matching approach

    The induced matching approach tries to create an unfiltered acyclic matching with small filtration differences
    and solves a minimal cut problem on this matching to get an induced filtered acyclic matching.

    Setting delta=0 results in the same algorithm as morse_reduce_nanda from MN[13]
    :param morse_complex: MorseComplex to reduce
    :param delta: Approximation parameter for the filtration and the persistence module
    :param return_matching: Return a tuple of the reduced complex and the matching used to reduce it
    :param deltatrick: Use 2*delta approximation from below and shift it down by delta. This allows more reductions.
    :returns Reduced complex, optionally also the filtered acyclic matching
    """
    delta = 2*delta if deltatrick else delta
    global_acyclic_matching = construct_acyclic_matching(morse_complex, delta)
    bins_aposteriori = find_discretization(global_acyclic_matching, morse_complex.filtration, delta)
    discrete_filtration = ceil(morse_complex.filtration, bins_aposteriori) - (deltatrick * delta / 2)
    filtered_acyclic_matching = global_acyclic_matching.induce_filtered_matching(discrete_filtration)
    reduced_cplx = reduce_by_acyclic_matching(filtered_acyclic_matching)
    if return_matching:
        return reduced_cplx, filtered_acyclic_matching
    return reduced_cplx


def morse_approx_along_gradients(morse_complex, delta, return_matching=False, deltatrick=True):
    """ Creates an approximative Morse reduction with the gradient approach

    The gradient approach prevents gradient paths to grow beyond a filtration difference larger than delta,
    therefore the resulting acyclic matching is properly filtered with respect to an induced filtration.
    :param morse_complex: MorseComplex to reduce
    :param delta: Approximation parameter for the filtration and the persistence module
    :param return_matching: Return a tuple of the reduced complex and the matching used to reduce it
    :param deltatrick: Use 2*delta approximation from below and shift it down by delta. This allows more reductions.
    If the trick is used this changes filtration values that lie outside the star of the reduced gradient paths.
    Can increase the bottleneck distance between true and approximated complex, but guarantees to stay below delta
    :returns Reduced complex, optionally also the filtered acyclic matching
    """
    delta = 2 * delta if deltatrick else delta
    acyclic_matching = construct_acyclic_matching_along_gradients(morse_complex, delta)
    # new_filtration <= morse_complex.filtration. So for the deltatrick we need to add.
    acyclic_matching.morse_complex.filtration = acyclic_matching.morse_complex.filtration + (deltatrick * delta / 2)
    reduced_cplx = reduce_by_acyclic_matching(acyclic_matching)
    if return_matching:
        return reduced_cplx, acyclic_matching
    return reduced_cplx


def reduce_until_stable(morse_complex):
    """ Reduces a complex until it stabilizes

    :param morse_complex:  A morse complex
    :return: A stably reduced morse complex"""
    while True:
        size = morse_complex.size
        morse_complex = morse_reduce(morse_complex)
        if size == morse_complex.size:
            return morse_complex


def morse_reduce(morse_complex, return_matching=False):
    """ Finds an acyclic matching and reduces by it

    Implements [MN13] p. 344 MorseReduce in a refactored version, untangling construction of an
    filtered acyclic matching from the reduction by it.

    Up to random effects it should hold that morse_reduce is the same as morse_reduce_nanda

    :param morse_complex: A morse complex
    :param return_matching: Return a tuple of the reduced complex and the matching used to reduce it
    :return: Pair of a reduced morse complex and the matches used to reduce it.
    The latter is returned for testing reasons.
    """
    acyclic_matching = construct_acyclic_matching(morse_complex, delta=0)
    reduced_cplx = reduce_by_acyclic_matching(acyclic_matching)
    if return_matching:
        return reduced_cplx, acyclic_matching
    return reduced_cplx


def unreduced_cells(akq, cell_ixs):
    """ Helper """
    return cell_ixs[np.where(akq[cell_ixs] == 0)[0]]


def unreduced_boundary(morse_complex, akq, cell_ix):
    """ Helper """
    return unreduced_cells(akq, morse_complex.get_boundary(cell_ix))


def unreduced_coboundary(morse_complex, akq, cell_ix):
    """ Helper """
    return unreduced_cells(akq, morse_complex.get_coboundary(cell_ix))


def relevant_cells(cell_ixs, relevant_ixs):
    return np.array(cell_ixs)[np.isin(cell_ixs, relevant_ixs)]


def construct_acyclic_matching(morse_complex, delta=np.inf):
    """ Finds an acyclic matching

    Inspired by [MN13] p. 344 MorseReduce

    :param morse_complex: A morse complex
    :param delta: Only construct matches with a filtration difference smaller than delta
    :return: Matching on morse_complex
    """
    morse_complex = morse_complex.copy().sort_by_filtration()
    reduced = np.full(morse_complex.size, False)  # Array to mark cells as reduced
    matching = Matching(morse_complex)
    while not np.all(reduced):
        ace_ix = new_ace_ix(reduced)
        queue = unreduced_coboundary(morse_complex, reduced, ace_ix).tolist()
        while queue:
            cell_ix = queue.pop(0)
            unreduced_cell_bd = unreduced_boundary(morse_complex, reduced, cell_ix)
            if len(unreduced_cell_bd) == 0:
                queue.extend(unreduced_coboundary(morse_complex, reduced, cell_ix))
            elif len(unreduced_cell_bd) == 1:
                q_ix = unreduced_cell_bd[0]
                if morse_complex.filtration[cell_ix] - morse_complex.filtration[q_ix] > delta:
                    continue
                add_matching((q_ix, cell_ix), matching, reduced)
                queue.extend(unreduced_coboundary(morse_complex, reduced, q_ix))
    return matching


def construct_acyclic_matching_along_gradients(morse_complex,
                                               delta=np.inf):
    """ Finds an acyclic matching in filtration order along gradients

    Inspired by [MN13] p. 344 MorseReduce

    :param morse_complex: A morse complex
    :param delta: Only construct matches with a filtration difference smaller than delta
    :return: Matching on morse_complex with approximate filtration.
    The approximate filtration will be less or equal than the old filtration.
    """
    new_filtration = ApproxFiltration(morse_complex.filtration.copy(),
                                      exact=morse_complex.filtration)
    morse_complex = morse_complex.copy(filtration=new_filtration).sort_by_filtration()
    matching = Matching(morse_complex)
    reduced = np.full(morse_complex.size, False)  # Array to mark cells as reduced
    while not np.all(reduced):
        grow_gradient_path(matching, reduced, delta)
    return matching


def grow_gradient_path(matching, reduced, delta):
    morse_complex = matching.morse_complex
    ace_ix = new_ace_ix(reduced)
    filtr_range = (morse_complex.filtration[ace_ix], morse_complex.filtration[ace_ix] + delta)
    queue = morse_complex.get_coboundary(ace_ix)
    while len(queue):
        relevant_ixs = relevant_indices(reduced, morse_complex.filtration, filtr_range)
        queue = np.unique(relevant_cells(queue, relevant_ixs)).tolist()
        if not queue: break
        cell_ix = queue.pop(0)
        relevant_cell_bd = relevant_cells(morse_complex.get_boundary(cell_ix), relevant_ixs)
        if len(relevant_cell_bd) == 0:
            queue.extend(morse_complex.get_coboundary(cell_ix))
        elif len(relevant_cell_bd) == 1:
            q_ix = relevant_cell_bd[0]
            add_matching((q_ix, cell_ix), matching, reduced)
            queue.extend(morse_complex.get_coboundary(q_ix))
            decrease_filtration(matching, q_ix, ace_ix)
            decrease_filtration(matching, cell_ix, ace_ix)


def add_matching(match, matching, reduced):
    """ Helper """
    reduced[match[1]] = True
    reduced[match[0]] = True
    matching.append(match)


def decrease_filtration(matching, cell_ix, target_ix):
    """ Decreases the filtration such that a given cell can get the given target's filtration value

    For this it decreases the filtration values of all the faces, if necessary """
    cell_filtration = matching.morse_complex.filtration[cell_ix]
    target_filtration = matching.morse_complex.filtration[target_ix]
    if target_filtration < cell_filtration:
        matching.morse_complex.filtration[cell_ix] = target_filtration
        for bdry_ix in matching.morse_complex.get_boundary(cell_ix):
            decrease_filtration(matching, bdry_ix, target_ix)


def new_ace_ix(reduced):
    """ Helper """
    # morse_complex is sorted by filtration, so this should have minimal filtration value
    result = np.where(reduced == False)[0][0]  # np.where returns tuple
    reduced[result] = True  # Mark the cell as ace, prior to update_gradients
    return result


def relevant_indices(reduced, filtration, filtration_range):
    """ Helper """
    return np.where((reduced == False) &
                    (min(filtration_range) <= filtration) &
                    (filtration <= max(filtration_range)))[0]


def reduce_by_acyclic_matching_dense(matching):
    """ Reduction of a complex by a given acyclic matching

    Assuming that the acyclic matching is correct, reducing by it is like a
    greedy persistent homology reduction ignoring the elder rule.

    This dense version is just for show cases and testing. Not optimized at all.

    :param matching: The acyclic matching in the form [(q1_ix, k1_ix), .. ]
    :return: The reduced morse complex
    """
    morse_complex = matching.morse_complex
    # An acyclic matching is like a greedy persistent homology reduction, ignoring the elder rule
    removed_pairs = np.full(morse_complex.size, False)  # Keeping track of the reduced cells
    boundary_matrix = np.array(morse_complex.boundary_matrix)
    for q_ix, k_ix in matching:
        # This could just be k_boundary = morse_complex.get_boundary(k_ix) if that would be efficient
        k_boundary = np.where(boundary_matrix[:, k_ix] != 0)[0]
        removed_pairs[q_ix] = True
        removed_pairs[k_ix] = True
        for q_coface_ix in np.where(boundary_matrix[q_ix, :] != 0)[0]:
            if removed_pairs[q_coface_ix]:
                continue
            q_coface_bdry_ixs = np.where(boundary_matrix[:, q_coface_ix] != 0)[0]
            new_coface_bdry_ixs = np.setxor1d(q_coface_bdry_ixs, k_boundary)
            new_boundary_row = np.full(morse_complex.size, False)
            new_boundary_row[new_coface_bdry_ixs] = True
            boundary_matrix[:, q_coface_ix] = new_boundary_row
    return morse_complex.copy(boundary_matrix=boundary_matrix)[matching.unmatched_ixs]


def reduce_by_acyclic_matching(matching):
    """ Complex resulting by reduction with a matching

    Assuming that the acyclic matching is correct, reducing by it is like a
    greedy persistent homology reduction ignoring the elder rule.
    :param matching: A Matching
    :return: The reduced morse complex
    """
    morse_complex = matching.morse_complex
    # An acyclic matching is like a greedy persistent homology reduction, ignoring the elder rule
    removed_pairs = np.full(morse_complex.size, False)  # Keeping track of the reduced cells
    # Scipy only implements row-wise LIL, to have fast column lookup and addition we need to transpose
    boundary_transposed = morse_complex.boundary_matrix_csr.tolil().transpose()
    for q_ix, k_ix in matching:
        k_boundary = boundary_transposed[k_ix, :].nonzero()[1]
        removed_pairs[q_ix] = True
        removed_pairs[k_ix] = True
        for q_coface_ix in boundary_transposed[:, q_ix].nonzero()[0]:
            if removed_pairs[q_coface_ix]:
                continue  # Dont update cells that are already removed
            coface_bdry_ixs = boundary_transposed[q_coface_ix, :].nonzero()[1]
            new_coface_bdry_ixs = np.setxor1d(coface_bdry_ixs, k_boundary)
            new_coface_bdry = np.full(morse_complex.size, False)
            new_coface_bdry[new_coface_bdry_ixs] = True
            boundary_transposed[q_coface_ix, :] = new_coface_bdry
    return morse_complex.copy(boundary_matrix=boundary_transposed.transpose())[matching.unmatched_ixs]
