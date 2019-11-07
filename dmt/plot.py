"""
Method to plot a MorseComplex based on points in 2D
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

from dmt.pers_hom import filter_zero_persistence_classes


def complex_with_points_2D(morse_complex=None, matching=None,
                           plot_filtration=[1, 2], filtration_fmt="%2.2f", fontsize=5,
                           point_size=3, cmap="plasma", linestyle="dashed", alpha=None,
                           arrowprops=dict(length_includes_head=True,
                                           width=0.005),
                           color_by_quantiles=False,
                           ax=None):
    """ Plots a complex with 2D points

    :param morse_complex: MorseComplex for plotting
    :param matching: Matching to be plotted
    :param plot_filtration: Add filtration values for the given dimensions to the plot
    :param filtration_fmt: Format string of the filtration values
    :param fontsize: Fontsize of the filtration values
    :param point_size: Size of the 0-cells
    :param cmap: Color map for the filtration values
    :param color_by_quantiles: Compute color by filtration quantiles instead of actual filtration value
    :param linestyle: Linestyle of the 1-cell
    :return: matplotlib axes

    :Example:
    >>> cplx = AlphaComplex(np.random.randn(10, 2))
    >>> complex_with_points_2D(cplx, matching=[(2,15), (3, 11)])
    >>> plt.show()
    """
    ax = plt.gca() if ax is None else ax
    if morse_complex is None and matching is None:
        raise ValueError("Either morse_complex or matching must not be None.")
    if matching:
        morse_complex = matching.morse_complex
    if morse_complex.points is None:
        raise ValueError("There are no points in morse_complex.points")
    if not morse_complex.points.shape[1] == 2:
        raise ValueError("Wrong dimension. Can only plot 2D points")
    shapes = []
    color_map = plt.get_cmap(cmap)
    color_filtration = morse_complex.filtration
    if color_by_quantiles:
        quantiles = np.unique(np.quantile(morse_complex.filtration, np.arange(0, 1.01, 0.01)))
        color_filtration = np.digitize(morse_complex.filtration, bins=quantiles, right=False) / len(quantiles)
    for simplex_ix in range(morse_complex.size):
        color = color_map(color_filtration[simplex_ix])
        dim = morse_complex.cell_dimensions[simplex_ix]
        if dim == 0:
            point = morse_complex.get_point(simplex_ix)
            shapes.append((point, point, simplex_ix, 0))
        elif dim == 1:
            point_ixs = morse_complex.get_boundary(simplex_ix)
            pts = morse_complex.get_point(point_ixs)
            if pts.shape[0] == 0:
                print("Found circle")  # Don't really know what to do, hope it still looks good
                continue
            center = np.sum(pts, axis=0) / pts.shape[0]
            shapes.append((Line2D(*pts.T, color=color, linestyle=linestyle, alpha=alpha),
                           center, simplex_ix, 1))
        elif dim == 2:
            # Need to sort the points along the order of the lines, so that the polygon has the correct shape
            lines = [morse_complex.get_boundary(line_ix) for line_ix in morse_complex.get_boundary(simplex_ix)]
            point_ixs = sorted_boundary(lines)
            if not point_ixs:
                continue
            pts = morse_complex.get_point(point_ixs)
            center = np.sum(pts, axis=0) / pts.shape[0]
            shapes.append((Polygon(pts, facecolor=color, edgecolor=None, alpha=alpha),
                           center, simplex_ix, 2))

    # Plot points with plt.scatter, this already sets the scale
    pts, pt_ixs, = zip(*[(p, s_ix) for (p, _, s_ix, s_dim) in shapes if s_dim == 0])
    pts = np.array(pts)
    pt_ixs = list(pt_ixs)
    ax.scatter(pts[:, 0], pts[:, 1], s=point_size, c=color_filtration[pt_ixs],
               cmap=color_map, zorder=5)
    adder_dict = {1: ax.add_line,
                  2: ax.add_patch}
    for shape_obj, center, simplex_ix, shape_dim in shapes:
        if shape_dim in plot_filtration:
            ax.text(center[0], center[1],
                     filtration_fmt % morse_complex.filtration[simplex_ix],
                     fontsize=fontsize)
        if shape_dim == 0:
            continue
        adder_dict[shape_dim](shape_obj)
    if matching is not None:
        for qix, kix in matching:
            _, start, _, _ = shapes[qix]
            _, end, _, _ = shapes[kix]
            diff = end - start
            ax.arrow(start[0], start[1], diff[0], diff[1], **arrowprops, zorder=5)
    return ax


def sorted_boundary(lines):
    """ Constructs a sorted boundary of a 2-cell

    :param lines: Assumes this is a list of arrays of length two that can be sorted in a closed boundary
    :returns: List """
    lines = [l for l in lines if len(l) == 2]
    if not lines:
        return []
    point_ixs = lines.pop(0).tolist()
    while point_ixs[-1] != point_ixs[0]:
        last_point = point_ixs[-1]
        for ix in range(len(lines)):
            if sum(lines[ix] == last_point) == 1:
                next_line = lines.pop(ix)
                break
        next_point = next_line[1] if next_line[0] == last_point else next_line[0]
        if next_point == point_ixs[-1]: break
        point_ixs.append(next_point)
    return point_ixs[:-1]  # First and last point are identical


def plot_diagrams(
            diagrams,
            title=None,
            dimensions=None,
            xy_range=None,
            colormap="default",
            size=20,
            ax_color=np.array([0.0, 0.0, 0.0]),
            diagonal=True,
            legend=True,
            plot_inf=True,
            plot_multiplicity=True,
            plot_zero_persistence_classes=False,
            label_fmt="$H_%i$",
            multiplicity_text_kwargs=dict(fontsize=7, horizontalalignment="right"),
            ax=None
    ):
    """A helper function to plot persistence diagrams adapted from persim
    """
    if isinstance(diagrams, dict):
        classes_filter = lambda x: x if plot_zero_persistence_classes else filter_zero_persistence_classes
        diagrams = [val for val in classes_filter(diagrams).values() if val.size]
    ax = ax or plt.gca()
    plt.style.use(colormap)

    xlabel, ylabel = "Birth", "Death"

    # Construct copy with proper type of each diagram
    # so we can freely edit them.
    diagrams = [dgm.astype(np.float32, copy=True) for dgm in diagrams]

    # find min and max of all visible diagrams
    concat_dgms = np.concatenate(diagrams).flatten()
    has_inf = np.any(np.isinf(concat_dgms))
    finite_dgms = concat_dgms[np.isfinite(concat_dgms)]

    # clever bounding boxes of the diagram
    if not xy_range:
        # define bounds of diagram
        ax_min, ax_max = np.min(finite_dgms), np.max(finite_dgms)
        x_r = ax_max - ax_min

        # Give plot a nice buffer on all sides.
        # ax_range=0 when only one point,
        buffer = 1 if xy_range == 0 else x_r / 5

        x_down = ax_min - buffer / 2
        x_up = ax_max + buffer

        y_down, y_up = x_down, x_up
    else:
        x_down, x_up, y_down, y_up = xy_range

    yr = y_up - y_down

    # Plot diagonal
    if diagonal:
        ax.plot([x_down, x_up], [x_down, x_up], "--", c=ax_color)

    # Plot inf line
    if plot_inf and has_inf:
        # put inf line slightly below top
        b_inf = y_down + yr * 0.95
        ax.plot([x_down, x_up], [b_inf, b_inf], "--", c="k", label=r"$\infty$")

        # convert each inf in each diagram with b_inf
        for dgm in diagrams:
            dgm[np.isinf(dgm)] = b_inf

    # Plot each diagram
    for dgm, dim in zip(diagrams, range(len(diagrams))):
        if dimensions and dim not in dimensions:
            continue
        # plot persistence pairs
        ax.scatter(dgm[:, 0], dgm[:, 1], size, label=label_fmt % dim, edgecolor="none")

    # Plot multiplicity
    if plot_multiplicity:
        for dgm in diagrams:
            unique_points = set(map(tuple, dgm.tolist()))
            for x, y in unique_points:
                count = sum((dgm[:, 0] == x) & (dgm[:, 1] == y))
                if count > 1:
                    # Lower right corner of the text is at x,y, as this moves it away from the diagnonal
                    ax.text(x, y, str(int(count)), **multiplicity_text_kwargs)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_xlim([x_down, x_up])
    ax.set_ylim([y_down, y_up])
    ax.set_aspect('equal', 'box')

    if title is not None:
        ax.set_title(title)

    if legend is True:
        ax.legend(loc="lower right")

    return ax
