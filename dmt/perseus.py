"""
Methods to wrap Perseus
https://people.maths.ox.ac.uk/nanda/perseus/index.html
"""
from time import time
import re
import os
from subprocess import check_output
from tempfile import mkstemp

import numpy as np
from scipy.spatial.distance import pdist

from dmt.binning import bin_cechmate


PERSEUS_EXECUTABLE = os.environ.get("PERSEUSPATH",
                                    os.path.join(os.path.dirname(__file__), "..", "Perseus", "perseus"))


def perseus_persistent_homology(morse_complex=None, cechmate_complex=None, delta=0, deltatrick=True):
    """ Compute Persistent Homology with Perseus

    :param morse_complex: Morse complex with a cechmate_complex field
    :param cechmate_complex: Simplicial complex in the format [(simplex, filtration),..],
    :param delta: Approximation parameter
    :param deltatrick: Use the delta trick
    :returns: Persistence diagram """
    if cechmate_complex is None:
        cechmate_complex = morse_complex.cechmate_complex
    if delta > 0:
        cechmate_complex = bin_cechmate(cechmate_complex, delta, deltatrick=deltatrick)
    tmp_fpath = get_tmp_fpath()
    step_filtration_dict = to_nmfsimtop(cechmate_complex, tmp_fpath)
    cmd = [PERSEUS_EXECUTABLE, "nmfsimtop", tmp_fpath, tmp_fpath]
    tic = time()
    perseus_out = check_output(cmd)
    print("Perseus total time (s): %s" % (time() - tic))
    print("Perseus reduced complex from %s to %s cells" % (len(cechmate_complex), parse_perseus_unreduced(perseus_out)))
    dgms = parse_perseus_homology(tmp_fpath, step_filtration_dict)
    cleanup(tmp_fpath)
    return dgms


def to_nmfsimtop(cechmate_cplx, filepath):
    """ Saves the complex in the nmfsimtop format of Perseus

    :param cechmate_cplx: Complex to save in cechmate format.
    :param filepath: Path to save to
    :returns: step_filtration_dict to translate Perseus' result back into the given filtration
    """
    simplices, filtration = zip(*cechmate_cplx)
    # Perseus expects filtration values to be integers > 0, -1 means np.inf
    unique, ixs = np.unique(filtration, return_index=True)
    ixs += 1
    filtration_step_dict = dict(zip(*[unique.tolist(), ixs.tolist()]))
    step_filtration_dict = dict(zip(*[ixs.tolist(), unique.tolist()]))
    step_filtration_dict[-1] = np.inf
    # Perseus wants all filtration values >0, so we add 1.
    lines = [' '.join(map(str, [len(simplex) - 1] + list(simplex) + [filtration_step_dict[filtr]]))
             for simplex, filtr in cechmate_cplx]
    with open(filepath, 'w') as fout:
        # Header gives ambient dimension, we just numerated all simplices
        fout.writelines('\n'.join(["1"] + lines))
    return step_filtration_dict


def get_tmp_fpath():
    """ Helper """
    tmp_file, tmp_fpath = mkstemp()
    os.close(tmp_file)
    return tmp_fpath


def cleanup(fpath):
    """ Cleanup files created by Perseus """
    os.remove(fpath)
    folder, base_fname = os.path.split(fpath)
    [os.remove(os.path.join(folder, fname)) for fname in os.listdir(folder)
     if fname.startswith(base_fname)]


def parse_perseus_homology(basepath, step_filtration_dict):
    """ Parse Perseus Files into a Persistence Diagram

    :param basepath: Basepath of the Perseus results, as given in the Perseus function call
    :param step_filtration_dict: How to translate the filtration steps of Perseus
    :returns: Persistence diagrams """
    folder, base_fname = os.path.split(basepath)
    files = [os.path.join(folder, fname) for fname in os.listdir(folder)
             if fname.startswith(base_fname) and fname.endswith(".txt") and fname[-5].isdigit()]
    return {parse_perseus_dim(fpath): parse_perseus_file(fpath, step_filtration_dict)
            for fpath in files}


def parse_perseus_file(fpath, step_filtration_dict):
    """ Parse a single perseus file into a persistence diagram """
    persistence = np.loadtxt(fpath, ndmin=2)
    for old, new in step_filtration_dict.items():
        persistence[persistence == old] = new
    return persistence


def parse_perseus_dim(fpath):
    """ Parse the dimension a Perseus file """
    folder, fname = os.path.split(fpath)
    return int(re.findall('_[0-9]+.txt', fname)[0][1:-4])


def perseus_unreduced_count(nmfsimtop_path):
    """ Counts the unreduced cells of Perseus

    Parses stdout after calling Perseus
    :param nmfsimtop_path: Path to an nmfsimtop file
    :returns: Count of critical cells
    """
    tmp_fpath = get_tmp_fpath()
    cmd = [PERSEUS_EXECUTABLE, "nmfsimtop", nmfsimtop_path, tmp_fpath]
    perseus_out = check_output(cmd)
    cleanup(tmp_fpath)
    return parse_perseus_unreduced(perseus_out)


def parse_perseus_unreduced(perseus_out):
    """ The remaining count of unreduced cells is the last digit in the output string """
    return [int(s) for s in perseus_out.split() if s.isdigit()][-1]


def save_points_perseus_brips(filepath, points):
    """ Saves points to Perseus' brips format

    :param filepath: Path to save to.
    :param points: Numpy array to save
    """
    dimension = points.shape[1]
    radius_scaling = 0.5
    step_size = 0.01
    steps = int(np.ceil(max(pdist(points)) / step_size))
    with open(filepath, 'w') as fout:
        print("Writing to file %s" % filepath)
        fout.writelines(["%i\n" % dimension,
                         "%s %s %s\n" % (radius_scaling, step_size, steps)] +
                        [("%s " * dimension + "0.0\n") % tuple(pt) for pt in points.astype(float)])


def load_points_perseus_brips(filepath):
    """ Loads points from Perseus' brips format

    :param filepath: Filepath to load from
    :returns: Numpy array of points
    """
    return np.loadtxt(filepath, skiprows=2)[:, :-1]
