#!/usr/bin/env python3
"""
Script to test the runtime of different approximation techniques
"""
import os
from datetime import datetime
from uuid import uuid4
import time
from itertools import product
from argparse import ArgumentParser

import pandas as pd

from dmt import morse_approx_binning, morse_approx_along_gradients, morse_approx_induced_matching
from dmt.data import load_complex, get_complex_names
from dmt.dmt import reduce_until_stable
from dmt.binning import bin_cechmate
from dmt.perseus import to_nmfsimtop, perseus_unreduced_count, get_tmp_fpath, cleanup


RESULTS_FOLDER = os.path.join(os.path.dirname(__file__), "..", "timings")
if not os.path.exists(RESULTS_FOLDER):
    os.mkdir(RESULTS_FOLDER)


def perseus_reduction(cplx, delta, prereduced_cplx=None, deltatrick=True):
    tmp_filename = get_tmp_fpath()
    binned_cechmate = bin_cechmate(cplx.cechmate_complex, delta, deltatrick=deltatrick)
    to_nmfsimtop(binned_cechmate, tmp_filename)
    unreduced_count = perseus_unreduced_count(tmp_filename)
    cleanup(tmp_filename)
    return cplx.size - unreduced_count


def reduction_wrapper(fn):
    def reduction_fn(cplx, delta, prereduced_cplx=None, deltatrick=True):
        original_size = cplx.size
        if prereduced_cplx is not None:
            cplx = prereduced_cplx
        return original_size - fn(cplx, delta, deltatrick=deltatrick).size
    return reduction_fn


binning_reduction = reduction_wrapper(morse_approx_binning)
gradient_reduction = reduction_wrapper(morse_approx_along_gradients)
induced_reduction = reduction_wrapper(morse_approx_induced_matching)
ALGO_MAP = {"binning": binning_reduction,
            "gradient": gradient_reduction,
            "induced": induced_reduction,
            "perseus": perseus_reduction
            }


def mytimeit(fun, *args, **kwargs):
    start = time.process_time()
    result = fun(*args, **kwargs)
    duration = time.process_time() - start
    return {"result": result,
            "time_s": duration}


def combine(dict1, dict2):
    """ update in place """
    dict1.update(dict2)
    return dict1


def compute_times(complex_fname, deltas, runs=3, algorithms=None):
    runs = list(range(runs))
    cplx = load_complex(complex_fname)
    prereduced_cplx = reduce_until_stable(cplx)
    algo_names = ALGO_MAP.keys() if algorithms is None else algorithms
    return pd.DataFrame([combine({"complex": str(type(cplx)),
                                 "filename": complex_fname,
                                 "run": uuid4().hex,
                                 "points": cplx.points.shape[0],
                                 "dim": cplx.points.shape[1],
                                 "size": cplx.size,
                                 "size_prereduced": prereduced_cplx.size,
                                 "delta": delta,
                                 "algorithm": algo_name},
                                 mytimeit(lambda: ALGO_MAP[algo_name](cplx, delta, prereduced_cplx=prereduced_cplx))
                                 )
                         for (algo_name, run, delta)
                         in product(algo_names, runs, deltas)])


def run_experiment(deltas, runs=5, complex_fnames=None, algorithms=None):
    times_dfs = []
    for complex_fname in complex_fnames or get_complex_names():
        print("Computing times for complex %s" % complex_fname)
        toc = time.time()
        times_df = compute_times(complex_fname, deltas, runs=runs, algorithms=algorithms)
        times_df.to_csv(times_fname(complex_fname))
        times_dfs.append(times_df)
        print("Done in %ss" % (time.time() - toc))
    return pd.concat(times_dfs)


def times_fname(complex_fname):
    return os.path.join(RESULTS_FOLDER, "times_%s_%s.csv" % (complex_fname, datetime.now()))


def load_times():
    return pd.concat([pd.read_csv(os.path.join(RESULTS_FOLDER, fname), index_col=0)
                      for fname in sorted(os.listdir(RESULTS_FOLDER))], ignore_index=True)


def get_parser():
    parser = ArgumentParser(description="Computing reduction sizes and runtimes "
                                        "of approximative Morse reduction")
    parser.add_argument("-d", "--deltas", help="Comma separated list of deltas",
                        type=lambda s: [float(item) for item in s.split(",")],
                        default=[0., 0.002, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.5, 1.])
    parser.add_argument("-a", "--algorithms", help="Comma separated list of algorithms to apply",
                        type=lambda s: [item for item in s.split(",")]
                        )
    parser.add_argument("-r", "--runs", help="Number of runs for each complex",
                        type=int, default=5)
    parser.add_argument("-c", "--complexes", help="Comma separated list of complexes",
                        type=lambda s: [item for item in s.split(",")]
                        )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    result = run_experiment(args.deltas, args.runs,
                            complex_fnames=args.complexes,
                            algorithms=args.algorithms)
    result["relative_reduction"] = result["result"] / result["size"]
    pd.options.display.max_columns = 20
    pd.options.display.max_rows = 100
    print(result.groupby(by=["algorithm", "delta", "size"]).describe()["relative_reduction"])


if __name__ == '__main__':
    main()
