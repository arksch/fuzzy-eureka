"""
Reimplementing [MN13] for testing that we do the same
[MN13] Konstantin Mischaikow and Vidit Nanda -
Morse Theory for Filtrations and Efficient Computation of Persistent Homology, 2013

"""
import numpy as np

from dmt.matching import Matching
from dmt.dmt import unreduced_coboundary, unreduced_boundary


def morse_reduce_nanda(morse_complex, return_matching=False):
    """ Finds an acyclic matching and reduces on the way

    Implements [MN13] p. 344 MorseReduce verbatim

    :param morse_complex: A morse complex
    :param return_matching: Return a tuple of the reduced complex and the matching used to reduce it
    :return: Pair of a reduced morse complex and the matches used to reduce it.
    The latter is returned for testing reasons.
    """
    morse_complex = morse_complex.sort_by_filtration()
    gradients = np.full((morse_complex.size, morse_complex.size), False)
    akq = np.zeros(morse_complex.size)  # Array to mark cells as aces and k-q matches
    matching = Matching(morse_complex)
    match_counter = 0
    for filtr_val in np.unique(morse_complex.filtration):
        cell_ixs = np.where(morse_complex.filtration == filtr_val)[0]
        unreduced_cell_ctr = len(cell_ixs)
        queue = list()
        while unreduced_cell_ctr > 0:
            ace_ix = cell_ixs[np.where(akq[cell_ixs] == 0)[0][0]]  # np.where returns tuple
            akq[ace_ix] = -1  # Mark the cell as ace, prior to update_gradients
            gradients = update_gradients(morse_complex, gradients, akq, ace_ix)
            unreduced_cell_ctr -= 1
            queue.extend(unreduced_coboundary(morse_complex, akq, ace_ix))
            while queue:
                cell_ix = queue.pop(0)
                if cell_ix not in cell_ixs:
                    continue
                unreduced_cell_bd = unreduced_boundary(morse_complex, akq, cell_ix)
                if len(unreduced_cell_bd) == 0:
                    queue.extend(unreduced_coboundary(morse_complex, akq, cell_ix))
                elif len(unreduced_cell_bd) == 1:
                    match_counter += 1  # Found a match, cell_ix is in K
                    q_ix = unreduced_cell_bd[0]
                    matching.append((q_ix, cell_ix))
                    akq[cell_ix] = match_counter
                    queue.extend(unreduced_coboundary(morse_complex, akq, q_ix))
                    if morse_complex.cell_dimensions[q_ix] == morse_complex.cell_dimensions[ace_ix]:
                        gradients[:, q_ix] = gradients[:, cell_ix]
                        gradients = update_gradients(morse_complex, gradients, akq, q_ix)
                    akq[q_ix] = match_counter
                    unreduced_cell_ctr -= 2
    ace_ixs = np.where(akq == -1)[0]
    reduced_cplx = morse_complex.copy(boundary_matrix=gradients)[ace_ixs]
    if return_matching:
        return reduced_cplx, matching
    return reduced_cplx


def update_gradients(morse_complex, gradients, akq, cell_ix):
    """ Keeps track of the discrete gradient paths when a cell is reduced to an ace or in Q

    Implements [MN13] p. 342 UpdateGradientChain
    :param morse_complex: Underlying complex
    :param gradients: Current state of the gradient paths
    :param akq: Array storing the reduced cells and their types (-1 entry being an ace)
    :param cell_ix: Index of the reduced cell in the morse complex
    :return: Updated gradient paths
    """
    for coboundary_cell_ix in unreduced_coboundary(morse_complex, akq, cell_ix):
        # x ^= y is shorthand for x = np.bitwise_xor(x, y)
        if akq[cell_ix] == -1:  # reduced cell is an ace
            gradients[cell_ix, coboundary_cell_ix] ^= 1
        else:  # reduced cell is in Q
            gradients[:, coboundary_cell_ix] ^= gradients[:, cell_ix]
    return gradients
