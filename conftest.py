""" Py.test configuration
https://stackoverflow.com/questions/34466027/in-pytest-what-is-the-use-of-conftest-py-files
"""
import pytest


def random_alpha_complex(samples, dim=2):
    import numpy as np
    from dmt.cechmate_wrap import AlphaComplex
    return AlphaComplex(np.random.randn(samples, dim))


@pytest.fixture
def random_alpha_complex_2D10_samples():
    return random_alpha_complex(10)


@pytest.fixture
def random_alpha_complex_2D50_samples():
    return random_alpha_complex(50)
