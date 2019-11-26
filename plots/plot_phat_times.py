"""
Plot how PHAT times evolve with different reductions
"""
from timeit import timeit
from uuid import uuid4
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from dmt.data import load_complex
from dmt import morse_approx_induced_matching, morse_approx_binning, morse_approx_along_gradients, morse_reduce


COMPLEX_FNAME = "normal_dist_2D_200pts_0.csv"
OUTPATH = "runtimes_phat_v2.pdf"
RECOMPUTE = False
DELTAS = [0.002, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2]
RUNS = 20

CPLX = None


def _get_cplx():
    """ Helper to access complexes in timeit setup"""
    return CPLX


def one_time(cplx, reduction, delta):
    global CPLX
    CPLX = cplx
    return {"reduction": reduction,
            "size": cplx.size,
            "delta": delta,
            "run": uuid4().hex,
            "PHAT_time": timeit("_get_cplx().persistence_diagram()", setup="from __main__ import _get_cplx", number=1)}


def compute_times(cplx, deltas):
    prereduced = morse_reduce(cplx)
    times = [one_time(cplx, "original", 0.) for i in range(RUNS)]
    times += [one_time(prereduced, "exact_morse", 0.) for i in range(RUNS)]
    algos = {"binning": morse_approx_binning,
             "induced": morse_approx_induced_matching,
             "gradient": morse_approx_along_gradients}
    for algo_name in algos:
        print(algo_name, )
        for delta in deltas:
            print(delta, )
            approx_cplx = algos[algo_name](prereduced, delta=delta, deltatrick=True)
            times += [one_time(approx_cplx, algo_name, delta) for i in range(RUNS)]
        print("")
    times = pd.DataFrame(times)
    return times


def plot(times):
    plot_df = times
    plot_df["reduction"][plot_df["reduction"] == "exact_morse"] = "exact_morse_reduce"
    plot_df["reduction"][plot_df["reduction"] == "original"] = "no_reduction"
    ax = sns.lineplot(x="delta", y="PHAT_time",
                       data=plot_df,
                       hue="reduction", ax=None)
    ax.set_ylabel("PHAT reduction time (sec)")
    ax = sns.scatterplot(x="delta", y="PHAT_time",
                      data=plot_df[plot_df["reduction"].isin(["no_reduction", "exact_morse_reduce"])],
                      hue="reduction", ax=ax, legend=None)


def main():
    if RECOMPUTE:
        cplx = load_complex(COMPLEX_FNAME)
        times = compute_times(cplx, DELTAS)
        times.to_csv("phat_times.csv")
    else:
        times = pd.read_csv("phat_times.csv")
    plot(times)
    plt.savefig(OUTPATH, bbox_inches="tight")
    plt.show()


if __name__ == '__main__':
    main()
