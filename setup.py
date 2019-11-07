#!/usr/bin/env python
import os
from setuptools import find_packages, setup


EXCLUDE_FROM_PACKAGES = []  # 'dmt.exclude_module'


def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname)) as f:
        return f.read()


setup(name='DMT',
      version='0.1',
      description='Discrete Morse theory for approximating persistent homology',
      long_description=read('README.md'),
      author='Arkadi Schelling',
      author_email='arkadi.schelling@gmail.com',
      license='BSD',
      packages=find_packages(exclude=EXCLUDE_FROM_PACKAGES),
      install_requires=['numpy', 'scipy', 'cechmate', 'persim', 'intervaltree']
      )
