# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development environment

Keep the virtual environment **outside the repository**, for example in
`~/python/venvs/pylorenzmie`. A venv inside the tree is swept by bulk edits
(`grep -r`, `sed`, `find`), which once rewrote a line of llvmlite inside one,
and `git clean -xdf` deletes it. Recreate it rather than move it, since venvs
contain absolute paths:

```bash
python3 -m venv ~/python/venvs/pylorenzmie
~/python/venvs/pylorenzmie/bin/pip install -e ".[dev,numba,docs]"
~/python/venvs/pylorenzmie/bin/python -m pytest
```

The `dev`, `numba` and `docs` extras are what development needs.  The `torch`, `jax`
and `gpu` extras are optional and not needed for the default test run; CI
installs `dev,numba` (and `dev` alone on one Python version). To check
the code without Numba, use a throwaway venv with `pip install -e ".[dev]"`.

## Commands

```bash
# Install in editable mode with test dependencies
pip install -e ".[dev,numba,docs]"

# Run all tests (the default run leaves out tests/slow and the JAX tests;
# python -m pytest -o addopts= runs them too)
python -m pytest

# Run a single test file
python -m pytest tests/test_lorenzmie.py

# Run a single test
python -m pytest tests/test_lorenzmie.py::TestLorenzMie::test_hologram

# Run with coverage report
python -m pytest --cov=. --cov-report=term-missing

# Lint (a CI job; configured in pyproject.toml)
ruff check .

# Type-check lib/ (a CI job; configured in pyproject.toml)
pyright

# Build HTML documentation (output: docs/_build/html/index.html)
sphinx-build -b html docs docs/_build/html
```

**Before a commit:** `pre-commit install` (with `pip install pre-commit ruff`)
runs ruff and the style, import-order, docstring, import-cycle and packaging
tests, the checks that otherwise fail late in CI.

## Media files

Media files (PNG, JPG, GIF, AVI, MP4) belong under `docs/`.  Two kinds are
part of the package and stay where they are: the test images in
`utilities/data/` and `lmtool/splash.png` (`.gitignore` lets them through).
Do not commit others elsewhere.

## Scratchpad directories

Any directory named `devel/` (at any depth) is a local scratchpad and is
ignored by git.  It exists only on the developer's machine and must never be
committed.  Scripts that make files shipped with the package, and that
someone must be able to run again, live in `tools/` instead (see its
`README.md`).

Each top-level directory has a short `README.md` that says what it is for;
update it when the role or contents of the directory change.

## Style

- Prefer single quotes over double quotes for strings, including docstrings.
- Maximum line length is 79 characters (PEP 8).
- Use American English spelling: *color*, *normalize*, *center*, *gray*, *behavior*.
- Layout of arguments, as in QVideo and QHOT.  In a definition, either
  everything fits on one line, or the parameters are aligned under the first,
  one to a line, with `self` or `cls` beside the first parameter.  In a call,
  either everything fits on one line, or the arguments are aligned under the
  first, one to a line, or they hang, one to a line with a trailing comma and
  the closing parenthesis on a line of its own (good for a call with
  keywords).  Never several arguments on a line of a wrapped list.
  `tests/test_style.py` enforces it.
- `ruff check .` (pyflakes, bugbear, the line length, single quotes, class
  attributes, quoted annotations, stale `noqa`, names that hide a builtin,
  parent imports and a few simplifications, configured in `pyproject.toml`)
  runs in CI as the `Lint` job: fix what it reports rather than silencing it.
  A `# noqa` says why.
- Annotations are not quoted; a module that needs a forward reference has
  `from __future__ import annotations` (ruff `UP037`).
- Every public class, function, method and property has a NumPy-style
  docstring.  Sphinx publishes only what has one, and
  `tests/test_docstrings.py` fails when one is missing (an override inherits
  the docstring of the method it overrides; a module that needs an optional
  library that is not installed is skipped).
- A parent package is never imported with dots (`from pylorenzmie.lib import
  LMObject`, not `from ..lib import LMObject`; ruff `TID252`).
  `tests/test_import_cycles.py` imports every module of every package first
  in an interpreter of its own (a module that needs an optional library that
  is not installed is skipped).  The order of the imports within a module is
  not checked.
- `pyproject.toml` lists every package; `tests/test_packaging.py` checks the
  list, because a package that is missing from it works in a checkout and is
  absent from the wheel.
- `pyright` checks everything but the backends that need Torch, Triton, JAX
  or CuPy (CI installs Numba and PyQt6 only).  Run it in an environment with
  exactly one Qt binding, PyQt6 (6.12.0: 6.12.1 types every `Signal(...)` as
  missing an argument), because pyright reads the stubs of a second one.  Fix
  what it reports rather than silencing it; a `# pyright: ignore` says why.
  A model that can scatter from a list of particles has no position of its
  own, so code that reads or sets one asks `model.single()` (or
  `model.single(Sphere)`), which raises a `TypeError` for a list.
- Docstrings use NumPy style with `Parameters`, `Returns`, `Raises`, and `Notes`
  sections as appropriate.
- Default values and units always noted in the Parameters section.
- All lengths are in **micrometers (μm)** except pixel coordinates (pixels).

## Releasing

`pyproject.toml` is the single source of truth for the version number.
`__init__.py` and `docs/conf.py` read it via `importlib.metadata` — do not
duplicate the version string anywhere else.
`CITATION.cff` is auto-updated by CI (`.github/workflows/update-citation.yml`)
on tag push — **never edit `CITATION.cff` manually**.

Release steps:

1. Bump `version` in `pyproject.toml` only.
2. Commit and push.
3. Tag and push the tag:

```bash
git tag vX.Y.Z && git push origin vX.Y.Z
```

The tag push triggers two CI workflows automatically:
- `publish.yml` builds the package, publishes to PyPI, and creates the
  GitHub Release (do not run `gh release create` manually).
- `update-citation.yml` updates `CITATION.cff` and commits it back.

The GitHub Release event fires Zenodo's webhook and mints a version DOI.
`CITATION.cff` carries the project (concept) DOI, 10.5281/zenodo.20784723,
which always resolves to the latest version, so nothing needs updating after
a release.

## Qt bindings

`lmtool` imports Qt through `qtpy` (`Signal`, `Slot`, `from qtpy import uic`),
never from `pyqtgraph.Qt` or a binding.  PyQt6, PyQt5 and PySide6 are tested
in CI (`pip install pylorenzmie[pyqt6]`, `[pyqt5]` or `[pyside6]`;
`[gui]` is `[pyqt6]`), and the environment variable `QT_API` chooses among
the ones that are installed.  `pyqtgraph` is a core dependency.

* Import `qtpy` before `pyqtgraph` in every module, script and example:
  `qtpy` honors `QT_API`, but `pyqtgraph` chooses a binding of its own when
  it is imported first, and `qtpy` then follows it.  A test file that does
  not import `lmtool` first has `import qtpy` ahead of `from pyqtgraph import
  mkQApp`.  `tests/test_qtpy_first.py` checks the text of every source file.
* A widget does not redefine a signal of its base class: PySide6 does not
  allow it.  `DoubleSlider` catches the change of its value in
  `sliderChange`, and `DoubleSpinBox` listens to its line edit.
* There are no Qt properties (`Property`): a property is a Python `property`.
* The stubs of the bindings type many getters as returning `None`;
  `pylorenzmie.lmtool.required.required` says that Qt made the object.  The
  widgets that `uic.loadUi` makes are declared as annotations on the class,
  with the names and classes of the `.ui` file.

## Architecture

The package computes and fits holograms of colloidal particles using Lorenz-Mie
scattering theory. The pipeline flows:
**generate coordinates → compute hologram → detect features → estimate
parameters → optimize fit**.

### `lib/` — Base infrastructure

**`LMObject`** is the abstract base class for every major class in the package.
It provides:
- A `properties` dict interface (get/set) used for serialization and fitting.
  Subclasses override the `properties` getter using `@ParentClass.properties.getter`.
  Only properties returned by `properties` are treated as adjustable parameters.
- JSON serialization (`to_json` / `from_json`).
- Pandas serialization (`to_pandas` / `from_pandas`).
- `meshgrid(shape, corner, flatten)` — generates pixel coordinate arrays of
  shape `(2, npts)` by default; pass `flatten=False` for `(2, ny, nx)`.

**`Azimuthal`** — module of functions (`avg`, `std`, `med`, `mad`) that compute
azimuthal statistics of 2-D images. They come from the `dgimage` package (the
image transformations that pylorenzmie shares with QVideo; edit them there, in
`~/python/dgimage`, not here): `lib/Azimuthal.py` only re-exports
them. Decorated with `@azimuthaloperator` so they accept `(data, center)` and
handle coordinate generation internally. Radial geometry is cached per
`(shape, center)`; the sort that only `med` and `mad` need is made lazily.

Other Python packages provide related functionality but do not substitute here:
- **pyFAI** — high-performance azimuthal integration for X-ray/neutron
  scattering; requires physical detector geometry and calibration, provides
  mean and variance only, no `med`/`mad`.
- **photutils** `RadialProfile` — concentric-annulus mean for astronomical
  point sources; no `std`, `med`, or `mad`.
- **scikit-image** `warp_polar` — remaps to polar coordinates (introduces
  interpolation); azimuthal mean requires a separate `.mean(axis=1)` step.
- **scipy.ndimage** — `ndimage.mean` matches `avg` but is marginally slower;
  `ndimage.median` is ~40% slower than the current loop-based `med`.

**`CircleTransform`** — ring-finding transform used by `Localizer` to detect
holographic ring patterns.  It comes from `dgimage` too (`lib/CircleTransform.py`
re-exports it, with the example).  The `Localizer` uses the default, double
precision transform; `dgimage` also offers `dtype=np.float32` and `binning=n`,
which trade accuracy for speed and are for a live display, not for fits.

### `theory/` — Physical models

**`Instrument`** — dataclass with microscope parameters: `wavelength` (μm),
`magnification` (μm/pixel), `n_m` (medium refractive index), `noise`. All four
appear in `properties` and are therefore adjustable during fitting.

**`Particle`** — abstract dataclass base for scatterers; holds 3D position
`r_p = [x_p, y_p, z_p]` and offset `r_0`. Implements `__len__`, `__iter__`,
`__next__`, and `__getitem__` so a single particle and a list behave uniformly
in `LorenzMie.field()`.

**`Sphere(Particle)`** — homogeneous sphere with `a_p` (radius, μm), `n_p`,
`k_p`. Computes Mie scattering coefficients `ab()` via the Wiscombe–Yang
algorithm.

**`Cluster`, `Dimer`** — multi-sphere assemblies.

**`LorenzMie(LMObject)`** — core scattering calculator (`method = 'numpy'`).
Takes `coordinates`, `particle`, and `instrument`; exposes `field()` and
`hologram()`. The class attribute `method` is used to match calculators with
compatible `Optimizer` instances.

**Accelerated subclasses** all inherit `LorenzMie` and override `_allocate()`
and `lorenzmie()`:
- `cupyLorenzMie` — CuPy (CUDA GPU), `method = 'cupy numpy'`
- `numbaLorenzMie` — Numba JIT, CPU; pointwise kernel, multi-threaded by
  default (`parallel = True`; fall back with `parallel = False` when the
  calling code is already parallel, see its docstring)
- `TorchLorenzMie` — GPU via PyTorch, `method = 'torch numpy'`: a Triton
  kernel on CUDA, a Metal kernel on Apple GPUs (MPS); single precision
  (hologram error about 1e-5 for a 0.5 um sphere, up to about 1e-3 for
  radii of a few um). On an M1 Pro it matches Numba for fits of 4,000
  pixels and is 2 to 3 times faster above about 25,000 pixels. The Metal
  kernel applies the `Aberrated` phase itself (the Triton path still
  multiplies by the mask in Python)
- `jaxLorenzMie` — JAX JIT with an analytical Jacobian; single spheres only
  (clusters fall back to NumPy) and it recompiles when the number of partial
  waves changes, so it is slower than Numba on a CPU
- `AberratedLorenzMie` — adds phase masks for the five primary Seidel
  aberrations (defocus, spherical, coma, astigmatism, distortion) and a
  constant `piston`; `from pylorenzmie.theory import AberratedLorenzMie` is
  built on the same calculator as `LorenzMie` (Numba if it works), and the
  NumPy one is `pylorenzmie.theory.Aberrated.AberratedLorenzMie`

`from pylorenzmie.theory import LorenzMie` is `numbaLorenzMie` when Numba works
and the NumPy class otherwise. The GPU backends and JAX are never chosen
automatically; select one with the environment variable
`PYLORENZMIE_BACKEND` (`numpy`, `numba`, `torch`, `jax` or `cupy`; an unavailable choice raises `ImportError`) or import
the class from its module.

### `analysis/` — Fitting pipeline

**`Localizer`** — detects ring-like features using `CircleTransform` + trackpy,
returns bounding boxes as `pandas.DataFrame` with columns `x_p`, `y_p`, `bbox`.

**`Estimator`** — estimates initial particle parameters (position, radius) from a
feature image using azimuthal analysis of the fringe pattern.

**`Mask`** — selects which pixels participate in the fit (excludes
saturated/NaN pixels; supports random subsampling via `fraction`: exactly
`round(fraction * n)` of the usable pixels, drawn from a private generator;
`seed` makes the draw reproducible).

**`Optimizer(LMObject)`** — wraps `scipy.optimize.least_squares`. The `method`
attribute must be compatible with the model's `method`. Properties `fixed` and
`variables` control which particle/instrument parameters are held constant vs.
optimized. Returns a `pandas.Series` with fitted values, uncertainties, and
reduced χ².

**`Feature`** — bundles a data array, coordinates, `Mask`, `LorenzMie` model,
`Estimator`, and `Optimizer` for a single particle. Call `estimate()` then
`optimize()`.

**`Frame(LMObject)`** — full-image pipeline. Call `frame.analyze(image)` to
run `detect() → estimate() → optimize()` on all features; returns a
`pandas.DataFrame` of results.

**`Trajectory`** — aggregates `Frame` results over time for particle tracking.

### `lmtool/` — GUI application

PyQtGraph-based interactive tool (`LMTool.py`). Loads hologram images, allows
interactive fitting, and displays results.

### `utilities/` — Helpers

Image normalization (`normalize_image.py`), running statistics
(`running_normal.py`, `vmedian.py`), water refractive index (`water.py`),
geometry utilities (`geometry.py`), visualization helpers (`visualization.py`).

## Test conventions

- Tests use `unittest`; the runner is `pytest`.
- All test images live in `tests/data/`. Reference them via
  `Path(__file__).parent / 'data' / 'filename.png'`.
- `# pragma: no cover` on all `if __name__ == '__main__':` guards.
- Every source directory is tested: `tests/test_tools.py` loads the scripts of
  `tools/` from their files (importing one must not do its work), and checks
  that the splash image that ships is what `tools/make_splash.py` makes.  A
  script is a `main()` and a guard, so that a test can import it.

## Documentation

Sphinx documentation uses the **PyData Sphinx Theme** (`pydata-sphinx-theme`)
with NYU brand colors applied via `docs/_static/nyu.css`:

- Primary: NYU purple `#57068c`
- Accent/hover: NYU violet `#8900e1`

`docs/conf.py` key settings:
- `html_theme = 'pydata_sphinx_theme'`
- Version read via `importlib.metadata.version('pylorenzmie')`
- `autodoc_mock_imports` covers `cupy`, `numba`, `torch`, `pyqtgraph`,
  `trackpy` (unavailable on doc-build hosts)
- NumPy-style docstrings via `sphinx.ext.napoleon`
