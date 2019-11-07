"""
A plot to show how the three algorithms reduce a complex
"""
import matplotlib.pyplot as plt

from dmt import morse_reduce, morse_approx_induced_matching, morse_approx_along_gradients, morse_approx_binning
from dmt.data import load_complex
from dmt.plot import complex_with_points_2D, plot_diagrams
from dmt.pers_hom import filter_zero_persistence_classes


EXAMPLE_COMPLEX = "normal_dist_2D_100pts_1.csv"
OUT_COMPLEX_PATH = "approximation_results.pdf"
OUT_DIAGRAMS_PATH = "approximation_diagrams.pdf"


def plot_scheme(cplx, matching, ax, title):
    complex_with_points_2D(cplx, matching=matching,
                           plot_filtration=[],
                           linestyle="solid",
                           point_size=0,
                           arrowprops=dict(length_includes_head=True,
                                           head_length=0.15,
                                           width=0.005)
                           )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=12)
    return ax


def plot_dgm(dgms, ax=None, delta=None, title=None):
    dgms = [val for val in filter_zero_persistence_classes(dgms).values() if val.size]
    plot_diagrams(dgms, size=5, legend=False, plot_inf=False, diagonal=False, ax=ax,
                  plot_multiplicity=True)
    # Plot diagnoal
    ax.plot([-10, 100], [-10, 100], "--", c="k", linewidth=.5)
    # Plot delta band
    if delta:
        ax.plot([-10, 100], [-10 + delta, 100 + delta], "--", c="k", linewidth=.5)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(title, fontsize=10)


def main():
    delta = 0.05
    deltatrick=True
    cplx = load_complex(EXAMPLE_COMPLEX)

    print("Loaded complex")
    reduced_cplx, filtered_matching = morse_reduce(cplx, return_matching=True)
    print("Prereduced complex")
    binned_cplx, binned_matching, reduced_binned = morse_approx_binning(reduced_cplx, delta,
                                                                        deltatrick=deltatrick,
                                                                        return_matching=True,
                                                                        return_binned_complex=True)
    print("Approximated with binning")
    induced_cplx, induced_matching = morse_approx_induced_matching(reduced_cplx.copy(), delta,
                                                                   deltatrick=deltatrick,
                                                                   return_matching=True)
    print("Approximated with induced")
    gradient_cplx, gradient_matching = morse_approx_along_gradients(reduced_cplx.copy(), delta,
                                                                    deltatrick=deltatrick,
                                                                    return_matching=True)
    print("Approximated with gradient")

    # ## Plotting the respective persistence diagrams
    reduced_dgm = reduced_cplx.persistence_diagram()
    binned_dgm = binned_cplx.persistence_diagram()
    induced_dgm = induced_cplx.persistence_diagram()
    gradient_dgm = gradient_cplx.persistence_diagram()
    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(141)
    plot_dgm(reduced_dgm, ax=ax, delta=delta, title="Original")
    ax = plt.subplot(142)
    plot_dgm(binned_dgm, ax=ax, delta=delta, title="Binning")
    ax.set_yticks([])
    ax = plt.subplot(143)
    plot_dgm(induced_dgm, ax=ax, delta=delta, title="Induced")
    ax.set_yticks([])
    ax = plt.subplot(144)
    plot_dgm(gradient_dgm, ax=ax, delta=delta, title="Gradient")
    ax.set_yticks([])
    plt.savefig(OUT_DIAGRAMS_PATH, bbox_inches="tight")
    plt.show()

    ### Plotting the complexes, matches and their reductions
    fig = plt.figure(figsize=(12, 8))
    ax = plt.subplot(421)
    plot_scheme(cplx, filtered_matching, ax, "Random Alpha complex with MorseReduce")
    ax = plt.subplot(422)
    plot_scheme(reduced_cplx, None, ax, "Result of exact MorseReduce, input for next steps")

    ax = plt.subplot(423)
    plot_scheme(reduced_binned, binned_matching, ax, "Prereduced complex with Binning-matching")
    ax = plt.subplot(424)
    plot_scheme(binned_cplx, None, ax, "Result of Binning")

    ax = plt.subplot(425)
    plot_scheme(reduced_cplx, induced_matching, ax, "Prereduced complex with Induced-matching")
    ax = plt.subplot(426)
    plot_scheme(induced_cplx, None, ax, "Result of Induced")

    ax = plt.subplot(427)
    plot_scheme(reduced_cplx, gradient_matching, ax, "Prereduced complex with Gradient-matching")
    ax = plt.subplot(428)
    plot_scheme(gradient_cplx, None, ax, "Result of Gradient")
    plt.savefig(OUT_COMPLEX_PATH, bbox_inches="tight")
    plt.show()


if __name__ == '__main__':
    main()
