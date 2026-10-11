# pylorenzmie

[![PyPI version](https://img.shields.io/pypi/v/pylorenzmie)](https://pypi.org/project/pylorenzmie/)
[![Python versions](https://img.shields.io/pypi/pyversions/pylorenzmie)](https://pypi.org/project/pylorenzmie/)
[![License](https://img.shields.io/pypi/l/pylorenzmie)](https://github.com/davidgrier/pylorenzmie/blob/master/LICENSE)
[![Documentation](https://readthedocs.org/projects/pylorenzmie/badge/?version=latest)](https://pylorenzmie.readthedocs.io)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20784723.svg)](https://doi.org/10.5281/zenodo.20784723)
[![Tests](https://github.com/davidgrier/pylorenzmie/actions/workflows/test.yml/badge.svg)](https://github.com/davidgrier/pylorenzmie/actions/workflows/test.yml)

Python routines for tracking and characterizing colloidal particles
with in-line holographic video microscopy (HVM)

## What pylorenzmie does
`pylorenzmie` provides a set of python classes for interacting with and
analyzing holographic microscopy data. The hologram of a colloidal
particle encodes comprehensive information about the particle's size,
composition, and location in three dimensions. This package extracts
that information by fitting a recorded hologram to a generative model
based on the Lorenz-Mie theory of light scattering.

<img src="https://raw.githubusercontent.com/davidgrier/pylorenzmie/master/docs/tutorials/crop.png" alt="Typical Hologram"
width="200"/>

## Documentation
The documentation is on [Read the Docs](https://pylorenzmie.readthedocs.io/en/stable/):

* the [quickstart](https://pylorenzmie.readthedocs.io/en/stable/quickstart.html)
  shows how to compute and fit holograms,
* the [API reference](https://pylorenzmie.readthedocs.io/en/stable/api/theory.html)
  describes the classes, and
* the [changelog](https://github.com/davidgrier/pylorenzmie/blob/master/CHANGELOG.md)
  lists what changed in each release and how to update.

The tutorial notebooks are rendered on GitHub:
[fitting a hologram](https://github.com/davidgrier/pylorenzmie/blob/master/docs/Fitting.ipynb),
[analyzing a frame](https://github.com/davidgrier/pylorenzmie/blob/master/docs/FrameAnalysis.ipynb),
[fitting a single frame](https://github.com/davidgrier/pylorenzmie/blob/master/docs/tutorials/SingleFrame.ipynb),
[holograms of several spheres](https://github.com/davidgrier/pylorenzmie/blob/master/docs/MultiSphere.ipynb),
[video analysis](https://github.com/davidgrier/pylorenzmie/blob/master/docs/Video_analysis.ipynb)
and [the building blocks of the model](https://github.com/davidgrier/pylorenzmie/blob/master/docs/tutorials/Theory.ipynb).
The documentation of the development version is
[here](https://pylorenzmie.readthedocs.io/en/latest/).

To report a problem or to contribute, see
[CONTRIBUTING.md](https://github.com/davidgrier/pylorenzmie/blob/master/CONTRIBUTING.md).

## Installation
`pylorenzmie` requires Python 3.11 or later:

```sh
pip install pylorenzmie
```

Optional extras add faster calculators:
`pip install pylorenzmie[numba]` (recommended: a fast CPU calculator),
`pylorenzmie[torch]` (GPUs, including Apple silicon), `pylorenzmie[gpu]`
(NVIDIA GPUs with CuPy), `pylorenzmie[jax]` and `pylorenzmie[gui]` (PyQt6, which
the `lmtool` application needs).
Extras can be combined, as in `pip install pylorenzmie[numba,torch]`.

## Example
Compute the hologram of a polystyrene sphere that is 1 micrometer in radius
and 200 pixels from the focal plane:

```python
from pylorenzmie.theory import LorenzMie, Sphere

model = LorenzMie()
model.coordinates = model.meshgrid((201, 201))
model.particle = Sphere(a_p=1., n_p=1.59, x_p=100., y_p=100., z_p=200.)
hologram = model.hologram().reshape(201, 201)
```

The [quickstart](https://pylorenzmie.readthedocs.io/en/stable/quickstart.html)
goes on to fit recorded holograms.

## Interactive particle tracking and characterization
`pylorenzmie` includes a GUI application called `lmtool`
for interacting with experimentally recorded holograms.

<img src="https://raw.githubusercontent.com/davidgrier/pylorenzmie/master/docs/lmtool/LMTool_ImageWidget.png" alt="LMTool image screen" width="30%"/>
<img src="https://raw.githubusercontent.com/davidgrier/pylorenzmie/master/docs/lmtool/LMTool_ProfileWidget.png" alt="LMTool profile screen" width="30%"/>
<img src="https://raw.githubusercontent.com/davidgrier/pylorenzmie/master/docs/lmtool/LMTool_FitWidget.png" alt="LMTool fit screen" width="30%"/>

## Authors
David G. Grier (New York University), Lauren Altman, Fook Chiong Cheong,
Mark D. Hannel, Sanghyuk Lee, Michael O'Brien, Sarah Odeh, Kaitlynn Snyder,
Jackie Sustiel

## Licensing.
[GPLv3](https://github.com/davidgrier/pylorenzmie/blob/master/LICENSE)

## References:
### Lorenz-Mie analysis of colloidal particles
1. S.-H. Lee, Y. Roichman, G.-R. Yi, S.-H. Kim, S.-M. Yang,
   A. van Blaaderen, P. van Oostrum and D. G. Grier,
   "Characterizing and tracking single colloidal particles with video
   holographic microscopy,"
   _Optics Express_ **15**, 18275-18282 (2007).
1. C. Martin, L. E. Altman, S. Rawat, A. Wang, D. G. Grier and V. N. Manoharan,
   "In-line holographic microscopy with model-based analysis,"
   _Nature Reviews Methods Primers_ **2**, 83 (2022).

### Lorenz-Mie theory of light scattering
1. C. F. Bohren and D. R. Huffman, Absorption and Scattering of Light
   by Small Particles (Wiley 1983).
1. M. I. Mishchenko, L. D. Travis and A. A. Lacis, Scattering
   Absorption and Emission of Light by Small Particles (Cambridge
   University Press, 2002).
1. G. Gouesbet and G. Gréhan, Generalized Lorenz-Mie Theories
   (Springer, 2011).

### Computational methods
1. W. Yang, "Improved recursive algorithm for light scattering
   by a multilayered sphere," _Applied Optics_ **42**, 1710--1720 (2003).
1. O. Pena and U. Pal, "Scattering of electromagnetic radiation
   by a multilayered sphere," _Computer Physics Communications_
   **180**, 2348-2354 (2009).
1. W. J. Wiscombe, "Improved Mie scattering algorithms,"
   _Applied Optics_ **19**, 1505-1509 (1980).
1. A. A. R. Neves and D. Pisignano, "Effect of finite terms on the
   truncation error of Mie series," _Optics Letters_ **37**,
   2481-2420 (2012).

## Acknowledgements
Work on this project at New York University is supported by the
National Science Foundation of the United States
under award number DMR-2428983.
