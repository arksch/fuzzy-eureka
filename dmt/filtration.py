"""
Filtration class
"""
import numpy as np


class Filtration(np.ndarray):
    """ Filtration """
    # https://docs.scipy.org/doc/numpy/user/basics.subclassing.html#basics-subclassing

    def __new__(cls, input_array):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        return obj


class ApproxFiltration(np.ndarray):
    """ Approximated Filtration """
    # https://docs.scipy.org/doc/numpy/user/basics.subclassing.html#basics-subclassing

    def __new__(cls, input_array, exact=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        obj.exact = Filtration(exact) if exact is not None else None
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.exact = getattr(obj, 'exact', None)

    def __getitem__(self, item):
        return ApproxFiltration(super().__getitem__(item),
                                self.exact[item] if self.exact is not None else None)

    @property
    def delta(self):
        return float(max(abs(self - self.exact)))
