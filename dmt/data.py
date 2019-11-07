"""
Example data
"""
import os

import numpy as np

from dmt import AlphaComplex


DATA_FOLDER = os.path.join(os.path.dirname(__file__), "..", "data")


def get_complex_names():
    return sorted(os.listdir(DATA_FOLDER))


def load_complex(fname, complex_class=AlphaComplex):
    filepath = os.path.join(DATA_FOLDER, fname)
    return complex_class(np.loadtxt(filepath))
