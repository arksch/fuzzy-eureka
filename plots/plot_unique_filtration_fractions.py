"""
Experiment to compute the fraction of unique filtration values for cells in an Alpha complex
"""
import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from dmt.complexes import AlphaComplex


RESULT_PATH = os.path.join(os.path.dirname(__file__), "reducible.csv")
PLOT_PATH = os.path.join(os.path.dirname(__file__), "reducible_fractions.pdf")


def get_points(point_count, dim):
    return np.random.random((point_count, dim))  # Only know asymptotics for uniform dist


def asymptotic_obtuseness_probability():
    """ Probability of a Delaunay triangle to be obtuse for uniformly distributed points
    https://math.stackexchange.com/a/2839303
    """
    return 0.5


def asymptotic_triangle_fraction():
    """
    Each triangle has three edges, each inner edge two triangles
    Boundary edges are rare, compared to inner edges.
    Therefore, the asymptotic ratio e:t is 3:2.
    Euler characteristic tells us that v+e-t=1, therefore, asymptotically v=0.5t
    """
    return 1./3


def asymptotic_reducible_cell_fraction():
    # Reductions depend only on the triangles, each reduction reduces two cells
    return 2 * asymptotic_triangle_fraction() * asymptotic_obtuseness_probability()


def create_complexes(point_counts, dim, repeat):
    cplx_dfs = []
    results = []
    for r in range(repeat):
        points = get_points(max(point_counts), dim)
        for point_count in point_counts:
            cplx = AlphaComplex(points[:point_count])
            cplx_df = pd.DataFrame(data=dict(filtration=cplx.filtration, dim=cplx.cell_dimensions))
            cplx_df = cplx_df.join(cplx_df.filtration.value_counts(), on="filtration", rsuffix="_count")
            cplx_df["point_count"] = point_count
            cplx_df["repeat"] = r
            cplx_dfs.append(cplx_df)
            results.append({"point_count": point_count,
                            "repeat": r,
                            "complex_size": len(cplx_df),
                            "0-cells": sum(cplx_df.dim == 0),
                            "1-cells": sum(cplx_df.dim == 1),
                            "2-cells": sum(cplx_df.dim == 2),
                            "reducible_cells": sum((cplx_df.dim > 0) & (cplx_df.filtration_count > 1)),
                            "reducible_2-cells": sum((cplx_df.dim == 2) & (cplx_df.filtration_count > 1))})
    results = pd.DataFrame(results)
    results["reducible_fraction"] = results["reducible_cells"] / results["complex_size"]
    results["reducible_2-cell_fraction"] = results["reducible_2-cells"] / results["2-cells"]
    results["2-cell_fraction"] = results["2-cells"] / results["complex_size"]
    complex_df = pd.concat(cplx_dfs)
    return results, complex_df


def plot_results(results):
    plot_df = pd.melt(results, id_vars=["point_count"],
                      value_vars=["2-cell_fraction", "reducible_fraction", "reducible_2-cell_fraction"],
                      value_name="Fraction",
                      var_name="Type")
    plot_df["Asymptotic"] = False
    point_counts = sorted(set(plot_df["point_count"]))
    plot_df = pd.concat([plot_df, pd.DataFrame({"point_count": point_counts,
                                                "Type": ["2-cell_fraction"] * len(point_counts),
                                                "Fraction": [asymptotic_triangle_fraction()] * len(point_counts),
                                                "Asymptotic": [True] * len(point_counts)})])
    plot_df = pd.concat([plot_df, pd.DataFrame({"point_count": point_counts,
                                                "Type": ["reducible_2-cell_fraction"] * len(point_counts),
                                                "Fraction": [asymptotic_obtuseness_probability()] * len(point_counts),
                                                "Asymptotic": [True] * len(point_counts)})])
    plot_df = pd.concat([plot_df, pd.DataFrame({"point_count": point_counts,
                                                "Type": ["reducible_fraction"] * len(point_counts),
                                                "Fraction": [asymptotic_reducible_cell_fraction()] * len(point_counts),
                                                "Asymptotic": [True] * len(point_counts)})])
    ax = sns.lineplot(x="point_count", y="Fraction", hue="Type", style="Asymptotic", data=plot_df)
    ax.set_ylim(-0.05, 1)
    ax.set_xlabel("Number of points")


def main():
    precomputed = True
    dim = 2
    point_counts = [10, 20, 50, 100, 200, 500, 1000, 2000]
    repeat = 50
    if precomputed:
        results = pd.read_csv(RESULT_PATH)
    else:
        results, complex_df = create_complexes(point_counts, dim, repeat)
        results.to_csv(RESULT_PATH)
    plot_results(results)
    plt.savefig(PLOT_PATH, bbox_inches="tight")


if __name__ == '__main__':
    main()
