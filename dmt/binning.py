"""
Binnings for approximations of filtrations
"""
import numpy as np
from intervaltree import Interval, IntervalTree

from dmt.filtration import ApproxFiltration


def get_binning(filtration, delta, deltatrick=True, epsilon=1e-9):
    """ Bin the filtration values by delta

    :param filtration: Filtration to be binned
    :param delta: Approximation parameter for the filtration
    :param epsilon: Dealing with right open intervals, shift delta grid by epsilon below lowest filtration value
    :param deltatrick: Use 2*delta approximation from below and shift it down by delta. This allows more reductions.
    :returns: Binned filtration values
    """
    if delta == 0:
        return filtration
    delta = delta + deltatrick * delta  # 2delta - delta is a better approximation
    # Make sure the minimal filtration values are not cut off into a separate bin.
    twodelta_bins = np.arange(min(filtration) - epsilon,
                              max(filtration) + delta,
                              delta)
    return ceil(filtration, twodelta_bins) - (deltatrick * delta / 2)


def bin_cechmate(cechmate_complex, delta, deltatrick=True):
    """ Bin a filtered simplicial complex in cechmate format

    :param cechmate_complex: Simplicial complex in the format [(simplex, filtration), ..]
    :param delta: Approximation parameter
    :param deltatrick: Use the deltatrick
    :returns: Binned cechmate complex
    """
    simplices, filtration = zip(*cechmate_complex)
    binned_filtration = get_binning(list(filtration), delta, deltatrick=deltatrick)
    return list(zip(simplices, np.asarray(binned_filtration)))


def ceil(filtration, discrete_values):
    """ Discretize a filtration to prespecified discrete values by ceiling


    :param filtration: A Filtration
    :param discrete_values: Discrete values for ceiling
    :returns: ApproxFiltration
    """
    discrete_values = sorted(discrete_values)
    new_values = np.append(discrete_values, np.inf)[np.digitize(filtration, bins=discrete_values, right=True)]
    return ApproxFiltration(new_values, exact=filtration)


def find_discretization(acyclic_matching, filtration, delta, epsilon=1e-9, return_partial_sols=False):
    """ Finds optimal cuts of a set of left open intervals

    This can be used to find an optimal induced filtered acyclic matching from an unfiltered one.
    Cover a set of intervals with cuts, such that
    - the first cut is before the first interval, the last cut after the last interval
    - consecutive cuts may not be further apart than delta
    - there are as few cuts as necessary

    Thanks to solution by stackoverflow user Orson L. Peters
    https://stackoverflow.com/questions/57250782/find-optimal-points-to-cut-a-set-of-intervals

    :param acyclic_matching: Pairs of filtration indices
    :param filtration: List with the filtration values of the matching
    :param delta: The maximal distance between two consecutive points of the discretization
    :param epsilon: Modeling open sets and dealing with inaccuracies of miniball
    :param return_partial_sols: Return the partial solutions, e.g for plotting
    :return: List with discrete points that optimally solve the interval cut problem
    """
    if delta == 0:
        return np.unique(filtration)
    # IntervalTree is set like. As our intervals might be exactly the same, we add some small noise
    rand_noise = np.random.uniform(0, epsilon, len(filtration))
    # Also, Alpha filtrations constructed by miniball are not exact and q might have a slightly higher value
    # Such degenerate cases would lead to an infinite loop below
    intervals = [Interval(filtration[q] - rand_noise[k], filtration[k]) for (q, k) in acyclic_matching
                 if filtration[q] < filtration[k]]
    if not intervals:
        return np.arange(min(filtration), max(filtration) + delta, delta)
    tree = IntervalTree(intervals)
    start = min(filtration) - epsilon  # Left open
    stop = max(filtration)
    # Go from left to right, store the best partial solution with k intervals containing a point.
    # We also store the intervals that these points are contained in as a set.
    furthest_point = start
    sol = [furthest_point]
    sols = {0: (sol, set())}
    partial_sols = [(sol, set())]
    while furthest_point < stop:
        lowest_k = min(sols.keys())
        sol, contained = sols.pop(lowest_k)
        furthest_point = sol[-1]
        extensions = find_right_changepoints(tree, furthest_point, furthest_point + delta, epsilon=epsilon)
        for ext in sorted(extensions, reverse=True):
            new_s = sol + [ext]
            new_contained = set(tree[ext]) | contained
            new_k = len(new_contained)
            if new_k not in sols or new_s[-1] > sols[new_k][0][-1]:
                sols[new_k] = (new_s, new_contained)
                partial_sols.append((new_s, new_contained))
    if return_partial_sols:
        return partial_sols
    return sol


def find_right_changepoints(tree, start, end, epsilon=1e-9):
    """ Find all change points in [start: end] where a left-open interval starts or ends

    :param tree: IntervalTree
    :param start: Point from which to look for change points
    :param end: How far to look for change points
    :param epsilon: Modeling open sets
    :returns All change points in tree[start: end]
    """
    # IntervalTree assumes right-open intervals, this makes indexing slightly unintuitive
    # tree = IntervalTree([Interval(0, 1)])
    # tree[0] >>> {Interval(0, 1)}
    # tree[1] >>> {}
    # tree[:0] >>> {}
    # tree[1:] >>> {}
    relevant_intervals = tree[start: end]
    relevant_points = [iv.begin - epsilon for iv in relevant_intervals]  # Left open
    relevant_points += [iv.end for iv in relevant_intervals]
    return {end} | {p for p in relevant_points if start < p < end}
