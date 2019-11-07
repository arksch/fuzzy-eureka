"""
Test if iterative reduction works
"""

import numpy as np
from dmt import AlphaComplex, morse_approx_induced_matching, morse_approx_along_gradients
from dmt.pers_hom import bottleneck


def iterative(cplx, delta, fn):
    """ Reduces a complex until it stabilizes """
    dgm = cplx.persistence_diagram()
    while True:
        size = cplx.size
        print("Size: %s" % size)
        cplx = fn(cplx, delta)
        print(bottleneck(dgm, cplx.persistence_diagram()))
        if size == cplx.size:
            return cplx


def main():
    cplx = AlphaComplex(np.random.randn(200, 2))
    for fn in [morse_approx_along_gradients, morse_approx_induced_matching]:
        print(fn)
        reduced = iterative(cplx, 0.05, fn)


if __name__ == '__main__':
    main()
