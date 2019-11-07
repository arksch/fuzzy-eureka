"""
Plotting reduction sizes
"""
import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


RESULTS_FOLDER = os.path.join(os.path.dirname(__file__), "..", "timings")
DELTAS = [0, 0.01, 0.05, 0.1]


def load_times():
    return pd.concat([pd.read_csv(os.path.join(RESULTS_FOLDER, fname), index_col=0)
                      for fname in sorted(os.listdir(RESULTS_FOLDER))], ignore_index=True)


def plot_reduction_over_delta(all_times, dim=2, max_delta=0.2, points=200):
    plot_df = all_times
    plot_df = plot_df[plot_df["dim"] == dim]
    plot_df = plot_df[plot_df["points"] == points]
    plot_df = plot_df[plot_df["delta"] <= max_delta]
    plot_df = plot_df[plot_df["algorithm"] != "perseus"]

    sns.lineplot(x="delta", y="relative_reduction", hue="algorithm", style="algorithm", data=plot_df)


def plot_reduction_over_points(all_times, dim=2, deltas=DELTAS):
    plot_df = all_times
    plot_df = plot_df[plot_df["dim"] == dim]
    plot_df = plot_df[plot_df["delta"].isin(deltas)]
    plot_df = plot_df[plot_df["algorithm"] != "perseus"]

    ax = sns.lineplot(x="points", y="relative_reduction",
                      hue="algorithm", style="delta",
                      data=plot_df, markers=True)
    ax.set_xticks(sorted(set(plot_df["points"])))


def plot_reduction_over_dim(all_times, points=200, deltas=DELTAS):
    plot_df = all_times
    plot_df = plot_df[plot_df["points"] == points]
    plot_df = plot_df[plot_df["delta"].isin(deltas)]
    plot_df = plot_df[plot_df["algorithm"] != "perseus"]

    ax = sns.lineplot(x="dim", y="relative_reduction",
                      hue="algorithm", style="delta",
                      data=plot_df, markers=True)
    ax.set_xticks(sorted(set(plot_df["dim"])))


def plot_runtimes_over_delta(all_times, dim=2, max_delta=0.2, points=200):
    plot_df = all_times
    plot_df = plot_df[plot_df["dim"] == dim]
    plot_df = plot_df[plot_df["points"] == points]
    plot_df = plot_df[plot_df["delta"] <= max_delta]
    # plot_df = plot_df[plot_df["algorithm"] != "perseus"]

    sns.lineplot(x="delta", y="time_s",
                 hue="algorithm", style="algorithm",
                 data=plot_df)


def plot_runtimes_over_dim(all_times, max_delta=0.2, points=200):
    plot_df = all_times
    plot_df = plot_df[plot_df["points"] == points]
    plot_df = plot_df[plot_df["delta"] <= max_delta]
    # plot_df = plot_df[plot_df["algorithm"] != "perseus"]

    sns.lineplot(x="dim", y="time_s", hue="algorithm", data=plot_df)


def plot_runtimes_over_points(all_times, deltas=DELTAS, dim=2):
    plot_df = all_times
    plot_df = plot_df[plot_df["dim"] == dim]
    plot_df = plot_df[plot_df["delta"].isin(deltas)]
    # plot_df = plot_df[plot_df["algorithm"] != "perseus"]

    ax = sns.lineplot(x="points", y="time_s",
                      hue="algorithm", style="delta", data=plot_df,
                      markers=True)
    ax.set_xticks(sorted(set(plot_df["points"])))


def print_results(all_times, points=200, dim=2):
    print_df = all_times
    print_df = print_df[print_df["dim"] == dim]
    print_df = print_df[print_df["points"] == points]
    print_df = print_df[print_df["algorithm"] != "perseus"]
    rel_red = print_df.groupby(by=["algorithm", "delta"]).describe()["relative_reduction"]

    print("\caption{Relative reduction sizes by algorithms and $\delta$ for %s points in $\R^%s$}" % (points, dim))
    print(rel_red.loc[["binning", "induced", "gradient"]][["min", "25%", "50%", "75%", "max", "mean", "std"]].to_latex(float_format="{:0.2f}".format))
    return rel_red


def plot_induced_advantage(all_times, points=200, dim=2):
    plot_df = all_times
    plot_df = plot_df[plot_df["dim"] == dim]
    plot_df = plot_df[plot_df["points"] == points]
    plot_df = plot_df[plot_df["algorithm"] != "perseus"]
    plot_grouped = plot_df.groupby(by=["algorithm", "delta", "filename"])["relative_reduction"].mean()
    binning_adv = plot_grouped.loc["induced"] - plot_grouped.loc["binning"]
    binning_adv = binning_adv.reset_index(level=0)
    binning_adv = binning_adv[binning_adv["delta"] > 0]
    ax = sns.lineplot(x="delta", y="relative_reduction", data=binning_adv)
    ax.set_xscale("log")
    ax.set_ylabel("r(Induced) - r(Binning)")
    return ax


def main():
    all_times = load_times()
    all_times["reduction"] = all_times["result"]
    all_times["relative_reduction"] = all_times["reduction"] / all_times["size"]

    rel_red = print_results(all_times)
    plot_induced_advantage(all_times)
    plt.savefig("induced_advantage.pdf", bbox_inches="tight")
    plt.show()

    plot_reduction_over_points(all_times, deltas=[0, 0.01, 0.05])
    plt.savefig("reduction_over_points.pdf", bbox_inches="tight")
    plt.show()

    plot_reduction_over_dim(all_times)
    plt.savefig("reduction_over_dim.pdf", bbox_inches="tight")
    plt.show()

    plot_reduction_over_delta(all_times, points=200)
    plt.savefig("reduction_over_delta_200pts_2D.pdf", bbox_inches="tight")
    plt.show()

    plot_reduction_over_delta(all_times, points=100)
    plt.savefig("reduction_over_delta_100pts_2D.pdf", bbox_inches="tight")
    plt.show()

    plot_runtimes_over_delta(all_times)
    plt.savefig("runtimes_over_delta.pdf", bbox_inches="tight")
    plt.show()

    plot_runtimes_over_dim(all_times)
    plt.savefig("runtimes_over_dim.pdf", bbox_inches="tight")
    plt.show()

    plot_runtimes_over_points(all_times, deltas=[0, 0.01])
    plt.savefig("runtimes_over_points.pdf", bbox_inches="tight")
    plt.show()


if __name__ == '__main__':
    main()
