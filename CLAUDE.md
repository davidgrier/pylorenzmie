# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install in editable mode with test dependencies
pip install -e ".[dev]"

# Install the post-merge git hook (one-time per clone); auto-updates
# ~/texmf/bibtex/bib/grier.bib after every git pull
ln -sf ../../scripts/update_bib.py .git/hooks/post-merge
chmod +x .git/hooks/post-merge

# Run all tests
python -m pytest

# Run a single test file
python -m pytest tests/test_lorenzmie.py

# Run a single test
python -m pytest tests/test_lorenzmie.py::TestLorenzMie::test_hologram

# Run with coverage report
python -m pytest --cov=. --cov-report=term-missing

# Build HTML documentation (output: docs/_build/html/index.html)
sphinx-build -b html docs docs/_build/html
```

## Style

- Prefer single quotes over double quotes for strings, including docstrings.
- Maximum line length is 79 characters (PEP 8).
- Use American English spelling: *color*, *normalize*, *center*, *gray*, *behavior*.
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

The GitHub Release event fires Zenodo's webhook and mints a DOI.

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
azimuthal statistics of 2-D images. Decorated with `@azimuthaloperator` so they
accept `(data, center)` and handle coordinate generation internally.

**`CircleTransform`** — ring-finding transform used by `Localizer` to detect
holographic ring patterns.

### `theory/` — Physical models

**`Instrument`** — dataclass with microscope parameters: `wavelength` (μm),
`magnification` (μm/pixel), `numerical_aperture`, `n_m` (medium refractive
index), `noise`. Only the first four appear in `properties` and are therefore
adjustable during fitting.

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
- `numbaLorenzMie` — Numba JIT, CPU
- `AberratedLorenzMie` — adds spherical aberration support

### `analysis/` — Fitting pipeline

**`Localizer`** — detects ring-like features using `CircleTransform` + trackpy,
returns bounding boxes as `pandas.DataFrame` with columns `x_p`, `y_p`, `bbox`.

**`Estimator`** — estimates initial particle parameters (position, radius) from a
feature image using azimuthal analysis of the fringe pattern.

**`Mask`** — selects which pixels participate in the fit (excludes
saturated/NaN pixels; supports random subsampling via `fraction`).

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
