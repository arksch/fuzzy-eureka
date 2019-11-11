#!/usr/bin/env python
import os
from setuptools import setup


def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname)) as f:
        return f.read()


setup(name='DMT',
      version='0.1',
      description='Discrete Morse theory for approximating persistent homology',
      long_description=read('README.md'),
      author='Arkadi Schelling',
      author_email='arkadi.schelling@gmail.com',
      url='https://github.com/arksch/fuzzy-eureka',
      license='MIT',
      packages=['dmt'],
      package_dir={'dmt': 'dmt'},
      install_requires=['numpy', 'scipy', 'cechmate', 'persim', 'intervaltree']
      )
