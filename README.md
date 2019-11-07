# Fuzzy Eureka

**Approximating Persistent Homology**

(With courtesy to github's repo name generator) 

Code base for a paper by Alex Wagner and Arkadi Schelling.

Extends ideas of Discrete Morse Theory from \[MN13\] *Konstantin Mischaikow and Vidit Nanda -
Morse Theory for Filtrations and Efficient Computation of Persistent Homology, 2013*



## Using the Code

Feel free to use this code for further development and research.
It is distributed under an MIT license.

### As a Library

The code is not packaged in pip.
If you are only planning to import this code, you can install it as a package.

Clone the repo
```
git clone https://github.com/arksch/fuzzy-eureka
cd fuzzy-eureka
```
Consider using a virtual environment
```
pip install virtualenv
virtualenv venv
venv/bin/activate
```
Install the `dmt` package (editable, so you can change the source)
```
pip install -r requirements.txt
pip install -e .
```

The fastest approximative algorithm is *Binning* with Perseus.
Install [Perseus](https://people.maths.ox.ac.uk/nanda/perseus/index.html),
then either put the executable at `./Perseus/perseus`
or set the environment variable `export PERSEUSPATH=/home/abc/Perseus/perseus`
This allows to run
```python 
import numpy as np
from dmt import AlphaComplex
from dmt.perseus import perseus_persistent_homology
dim = 2
samples = 1000
delta = 0.01
cplx = AlphaComplex(np.random.randn(samples, dim))
diagrams = perseus_persistent_homology(cplx, delta=delta)
```

### Plotting

If you are planning to use the plotting abilities of this repo, you should also run
```
pip install -r requirements.plot.txt
```

Apart from methods in `dmt.plot` to plot persistence diagrams and
cell complexes based on 2D point clouds this also allows to run the 
scripts from [plot](plot/), e.g.
```
python plot/plot_three_approximations.py
```
to create the following image

![Figure1](plots/approximation_results.png)

### Further Development

Install [Gudhi](https://gudhi.inria.fr/python/latest/installation.html),
which is not packaged in pip (but with conda).
We need Gudhi only for testing, as it computes the correct Bottleneck distances.
By the time of writing persim's Bottleneck was buggy.

Run tests with
```
py.test -v
```

Create HTML code coverage report at `htmlcov/index.html` that can be opened in your browser
```
py.test --cov=.
coverage html
```
