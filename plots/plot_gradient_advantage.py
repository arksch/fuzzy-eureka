"""
Create plots for the paper
"""
import matplotlib.pyplot as plt

from dmt import MorseComplex, morse_approx_along_gradients, morse_approx_induced_matching
from dmt.plot import complex_with_points_2D


OUT_PATH = "gradient_advantage.pdf"


def get_gradient_advantage_complex():
    # A very basic example where it is better to have a general approximation than a global approximation
    gradient_advantage_cplx = MorseComplex(boundary_matrix=[[0, 0, 0, 0, 1, 0],
                                                            [0, 0, 0, 0, 1, 0],
                                                            [0, 0, 0, 0, 0, 1],
                                                            [0, 0, 0, 0, 0, 1],
                                                            [0, 0, 0, 0, 0, 0],
                                                            [0, 0, 0, 0, 0, 0]],
                                           cell_dimensions=[0, 0, 0, 0, 1, 1],
                                           filtration=[0, 0, .5, .5, 1, 1.5],
                                           points=[[0, 0], [1, 0], [0, 1], [1, 1]])
    return gradient_advantage_cplx


def plot_scheme(cplx, matching, ax):
    ax.set_aspect(.3)
    complex_with_points_2D(cplx, matching=matching,
                           plot_filtration=[0, 1],
                           fontsize=10,
                           linestyle="solid",
                           cmap="Dark2",
                           alpha=.8,
                           filtration_fmt="%1.1f",
                           point_size=6,
                           arrowprops=dict(length_includes_head=True,
                                           head_length=0.15,
                                           width=0.04,
                                           color="purple")
                           )
    ax.set_ylim([-.2, 1.4])  # Text appears above
    ax.set_xlim([-.3, 1.3])
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def main():
    delta = .5 + 1e-6  # Need to add some delta to properly deal with the open intervals.
    gradient_advantage_cplx = get_gradient_advantage_complex()
    gradient_reduced, gradient_matching = morse_approx_along_gradients(gradient_advantage_cplx, delta,
                                                                       return_matching=True, deltatrick=True)
    induced_reduced, induced_matching = morse_approx_induced_matching(gradient_advantage_cplx, delta,
                                                                      return_matching=True, deltatrick=True)
    ax0 = plt.subplot(131)
    ax0 = plot_scheme(gradient_advantage_cplx, None, ax0)
    ax0.set_title("Original complex", fontsize=10)
    ax1 = plt.subplot(132)
    ax1 = plot_scheme(gradient_advantage_cplx, gradient_matching, ax1)
    ax1.set_title("Gradient Algorithm", fontsize=10)
    ax2 = plt.subplot(133)
    ax2 = plot_scheme(gradient_advantage_cplx, induced_matching, ax2)
    ax2.set_title("Induced Algorithm", fontsize=10)
    plt.savefig(OUT_PATH, bbox_inches="tight")


if __name__ == "__main__":
    main()
