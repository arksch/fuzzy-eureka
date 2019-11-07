"""
Example where iterative delta-approximations increase the Bottleneck distance arbitrarily
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from dmt import MorseComplex
from dmt.dmt import reduce_by_acyclic_matching
from dmt.matching import Matching
from dmt.plot import complex_with_points_2D, plot_diagrams


OUT_PATH = "concentric_annuli.pdf"


def square_points():
    return np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])


def square_boundary():
    return np.array([[0, 0, 0, 0, 1, 0, 0, 1,],
                     [0, 0, 0, 0, 1, 1, 0, 0,],
                     [0, 0, 0, 0, 0, 1, 1, 0,],
                     [0, 0, 0, 0, 0, 0, 1, 1,],
                     [0, 0, 0, 0, 0, 0, 0, 0,],
                     [0, 0, 0, 0, 0, 0, 0, 0,],
                     [0, 0, 0, 0, 0, 0, 0, 0,],
                     [0, 0, 0, 0, 0, 0, 0, 0,]])


def combine_boundary_matrices(A, B):
    return np.block([[A, np.zeros((A.shape[0], B.shape[1]))],
                     [np.zeros((B.shape[0], A.shape[1])), B]])


def add_cell(boundary_matrix, faces):
    new_column = np.zeros((boundary_matrix.shape[0], 1))
    for face in faces:
        new_column[face] = 1
    boundary_matrix = np.block([[boundary_matrix, new_column],
                                [np.zeros((1, boundary_matrix.shape[1] + 1))]])
    return boundary_matrix


def central_square(max_filtration):
    max_filtration = float(max_filtration)
    points = square_points()
    boundary_matrix = square_boundary()
    boundary_matrix = add_cell(boundary_matrix, [4, 5, 6, 7])
    cell_dimensions = [0, 0, 0, 0, 1, 1, 1, 1, 2]
    filtration = [max_filtration - 1] * 8 + [max_filtration]
    assert MorseComplex(boundary_matrix=boundary_matrix,
                        cell_dimensions=cell_dimensions,
                        filtration=filtration,
                        points=points).valid_boundary()
    matches = [(4, 8)]
    return points, boundary_matrix, cell_dimensions, filtration, 0, matches


def add_annulus(points, boundary_matrix, cell_dimensions, filtration, prev_base, matches):
    base = boundary_matrix.shape[0]
    points = np.concatenate([points, square_points() * (points.max() + 1)])
    boundary_matrix = combine_boundary_matrices(boundary_matrix, square_boundary())
    prev_square_base = prev_base + 4
    square_base = base + 4
    ## Add four diagonal edges counter clockwise
    diagonal_base = square_base + 4
    for i in range(4):
        boundary_matrix = add_cell(boundary_matrix, [prev_base + i, base + i])  # this is diagonal_base + i
    ## Four trapecoids counter clockwise
    trapecoid_base = diagonal_base + 4
    for i in range(4):
        trapecoid = [prev_square_base + i, square_base + i, diagonal_base + i, diagonal_base + ((i + 1) % 4)]
        boundary_matrix = add_cell(boundary_matrix, trapecoid)
        matches.append((square_base + i, trapecoid_base + i))
    cell_dimensions += [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2]
    filtration += [min(filtration) - 1] * 8 + [min(filtration)] * 8
    assert MorseComplex(boundary_matrix=boundary_matrix,
                        cell_dimensions=cell_dimensions,
                        filtration=filtration,
                        points=points).valid_boundary()
    return points, boundary_matrix, cell_dimensions, filtration, base, matches


def concentric_annuli_matching(n=3):
    max_filtration = n + 1
    points, boundary_matrix, cell_dimensions, filtration, base, matches = central_square(max_filtration)
    for i in range(n):
        points, boundary_matrix, cell_dimensions, filtration, base, matches = add_annulus(points, boundary_matrix, cell_dimensions, filtration, base, matches)
    gradient_advantage_cplx = MorseComplex(boundary_matrix=boundary_matrix,
                                           cell_dimensions=cell_dimensions,
                                           filtration=filtration,
                                           points=points,
                                           sort_by_filtration=False)
    assert gradient_advantage_cplx.valid_boundary()
    matching = Matching(morse_complex=gradient_advantage_cplx,
                        matches=matches)
    return matching


def plot(matching):

    def plot_diags(diags, ax=None):
        ax = plot_diagrams(diags,
                           legend=False,
                           dimensions=[1],
                           plot_multiplicity=False,
                           plot_inf=False,
                           plot_zero_persistence_classes=True,
                           xy_range=(-0.5, 5.5*.95, -0.5, 5.5*.95),  # Infty on 5.
                           ax=ax)
        ax.set_yticks([0, 2, 4])
        ax.set_xticks([0, 2, 4])
        ax.set_ylabel("")
        ax.set_xlabel("")
        return ax


    plt.subplot(131)
    ax = complex_with_points_2D(matching=matching,
                                plot_filtration=[0,1,2],
                                color_by_quantiles=True,
                                filtration_fmt="%i")
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_aspect('equal', 'box')
    ax.set_title("Annuli with matching", fontsize=8)

    plt.subplot(132)
    reduced_cplx = reduce_by_acyclic_matching(matching)
    ax = complex_with_points_2D(morse_complex=reduced_cplx,
                                plot_filtration=[1, 2],
                                color_by_quantiles=True,
                                filtration_fmt="%i")
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_aspect('equal', 'box')
    ax.set_title("Reduced annuli", fontsize=8)

    plt.subplot(133)
    ax = plot_diags(matching.morse_complex.persistence_diagram())
    ax.set_title("$H_1$ of annuli", fontsize=8)



def main():
    matching = concentric_annuli_matching()
    plot(matching)
    plt.savefig(OUT_PATH, bbox_inches="tight")
    plt.show()


if __name__ == '__main__':
    main()
