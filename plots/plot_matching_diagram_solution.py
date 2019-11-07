"""
Plotting how an optimal solution to a matching diagram is found
"""
import numpy as np
import matplotlib.pyplot as plt

from dmt.binning import ceil, find_discretization


OUT_FILE = "matching_diagram_solution.pdf"


def get_matching(delta):
    """
    Some minimal example that still is interesting, e.g.
      –––
      –––
    ––– –––
        –––
    """
    length = .95 * delta
    gap = .2 * delta
    start0, end0 = 0, length
    start1, end1 = .5 * length, 1.5 * length
    start2, end2 = length + gap, 2 * length + gap
    matching = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    filtration = [start0, end0, start1, end1, start1, end1, start2, end2, start2, end2]
    return matching, filtration


def plot_partial_sol(matching, filtration, partial_sol, delta, ax=None, plot_xticks=False, xlim=None):
    points, intervals = partial_sol
    ceiled_filtration = ceil(filtration, points)
    ax = ax or plt.gca()
    ax.set_aspect(.3)
    height = .2*(max(filtration) - min(filtration)) / len(matching)
    for h, match in zip(range(len(matching)), matching):
        s, e = match
        begin, end = filtration[s], filtration[e]
        is_seen = begin <= points[-1]
        is_cut = is_seen and ceiled_filtration[s] != ceiled_filtration[e]
        linestyle = "solid" if is_seen else "dashed"
        color = "red" if is_cut else "k"
        ax.plot([begin, end], [h * height, h * height],
                linestyle=linestyle, c=color, linewidth=1)
    for point in points[:-1]:
        ax.axvline(point, linestyle="--", c="grey")
    ax.axvline(points[-1], linestyle="--", c="k")

    ax.set_ylabel("%s" % len(intervals))
    ax.set_ylim([-height, (len(matching) + .5) * height])
    ax.set_yticks([])
    ax.set_xticks([])
    if plot_xticks:
        xticks = np.arange(0, (max(filtration) + delta) / delta, 1)
        ax.set_xticks(xticks * delta)
        ax.set_xticklabels(["%i$\delta$" % i for i in xticks])
    if xlim:
        ax.set_xlim(xlim)


def plot_solving(matching, filtration, partial_sols, delta):
    partial_sols = partial_sols[:-1]
    steps = len(partial_sols)
    xlim = [-.2, 2.2]
    for k in range(steps - 1):
        ax = plt.subplot(int("%i1%i" % (steps, k + 1)))
        plot_partial_sol(matching, filtration, partial_sols[k], delta, ax=ax, plot_xticks=False, xlim=xlim)
    k = steps - 1
    ax = plt.subplot(int("%i1%i" % (steps, k + 1)))
    plot_partial_sol(matching, filtration, partial_sols[k], delta, ax=ax, plot_xticks=True, xlim=xlim)


def main():
    delta = 1
    matching, filtration = get_matching(delta)
    partial_sols = find_discretization(matching, filtration, delta, return_partial_sols=True)
    plot_solving(matching, filtration, partial_sols, delta)
    plt.savefig(OUT_FILE, bbox_inches="tight")


if __name__ == '__main__':
    main()
