# Changelog

All notable changes to pylorenzmie are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [Unreleased]

### Added

* `lmtool` looks for particles when it loads an image and puts the ROI on the
  first, sized to its fringes. **Particles -> Next Particle** (Ctrl+N) selects
  the others and **Find Particles** (Ctrl+L) searches again.
* `lmtool` reports the reduced chi-squared of a fit in the status bar, and
  background tasks that fail log their traceback and show the exception type.

### Changed

* `lmtool`: the model in the Fit tab is no longer clipped to the range of the
  data, so the residuals show the whole difference.
* `lmtool`: spherical aberration is held fixed in fits until you free it, like
  the piston.

### Fixed

* `lmtool`: saving a result raised a `TypeError` with pandas 3, which made
  `key` a keyword-only argument, and the error was lost. Saving now reports its
  success or failure in the status bar, writes JSON when PyTables is not
  installed, creates `~/data/lmtool` if it is missing, and treats any name
  that does not end in `.json` as HDF5 in *Save Result As*. The `gui` extra
  installs PyTables.

## [3.1.1] - 2026-10-11

A fix for fits in `lmtool`, and a new layout of its second row of controls.

### Changed

* `lmtool`: the second row of the instrument controls holds noise, piston and
  spherical aberration, which fills the cell that the removed numerical-aperture
  control left empty. The piston is held fixed in fits until you free it.

### Fixed

* `lmtool` freed every model parameter that has no control in the panel. With
  the aberrated model that meant defocus, coma, astigmatism, distortion and the
  aberration center, which a fit moved far from the solution (defocus is nearly
  degenerate with the height, and the center is meaningless without the terms
  that use it), and the next estimate, made with the model left in that state,
  was nonsense. Parameters without a control are now held fixed.

## [3.1.0] - 2026-10-11

A phase offset for the scattered field, a license file, a `gui` extra, a
correction to the density of water, and more tests and documentation.

### Added

* `Aberrated(...)` classes take a `piston` parameter: a constant phase of the
  scattered field relative to the reference field, in wavelengths like the other
  coefficients. It is not a geometric aberration (it has no generating
  function), it is zero by default, and it works in all calculators, including
  the Metal and JAX kernels. Treat a fitted piston as a diagnostic: it is
  nearly degenerate with the particle's refractive index.
* A `LICENSE` file (GPLv3), included in the wheel and the source distribution.
* A `gui` extra (`pip install pylorenzmie[gui]`) that installs PyQt6, which
  `lmtool` needs. Previously a plain install had no Qt binding.
* `py.typed`, so type checkers use the package's annotations.
* `CONTRIBUTING.md` and documentation pages on the architecture, the
  conventions and how to extend the package.
* Tests for the utilities, the properties protocol, backend fallbacks,
  `Optimizer.report()` and the lmtool widgets (run offscreen in their own CI
  job). The CI coverage job requires 62%.

### Fixed

* `utilities.water.density()` had the wrong sign on the cubic term of Kell's
  formula, a transposed digit and a stray factor. It was in error by 2% at 80
  celsius and affected `refractiveindex()` (3e-4 at 25 C, 1.6e-3 at 40 C). It
  now agrees with IAPWS-95 to 0.01 kg/m^3.
* The quickstart showed three of its sections twice.
* Array annotations use `np.floating`, `np.complexfloating` and `np.integer`.

## [3.0.0] - 2026-10-11

Faster calculators, GPU support on Apple silicon, and a set of corrections to
defaults that made fits non-reproducible or silently wrong. Several of them
change behavior, so this is a major release; see *Breaking changes* for how to
update.

### Breaking changes

* **Python 3.11 or later is required.** Python 3.10 is no longer supported.
  Versions 3.11 to 3.14 are tested.
* **`pylorenzmie.theory.LorenzMie` is now the fastest calculator that needs no
  special hardware:** `numbaLorenzMie` if Numba is installed and works,
  otherwise the NumPy calculator. It used to prefer JAX, then CuPy, then Numba.
  JAX, CuPy and torch are never chosen automatically. To use one, set the
  environment variable `PYLORENZMIE_BACKEND` to `numpy`, `numba`, `torch`, `jax`
  or `cupy` before importing pylorenzmie, or import the class from its module.
  An unavailable choice raises `ImportError`; an unknown name raises
  `ValueError`.
* **`Instrument.numerical_aperture` is removed.** Nothing used it.
  `Instrument(numerical_aperture=...)` raises `TypeError`, and positional
  arguments after `magnification` shift by one. Saved property dictionaries
  that still contain the key load normally; the key is ignored.
* **`Localizer` bounding boxes are sized differently.** A fringe is now one
  bright and one dark ring (two crossings of the radial profile through 1), and
  the box extends from the center to the radius of the `2 * nfringes`-th
  crossing. Boxes used to be about half that size. The default `nfringes` is 4
  instead of 20, which gives a box close to the old default's in size
  (191 against 171 pixels in a test case). Boxes are clipped to the image, so
  one at the edge is not square, bounding-box entries are Python integers, and
  an image without features returns an empty table with the columns `x_p`,
  `y_p` and `bbox`. Re-tune any code that depended on the old meaning of
  `nfringes`.
* **`DEEstimator` searches in one process by default** (`settings['workers']`
  is 1 instead of -1). Starting a pool of worker processes costs seconds on
  macOS and Windows and made the search several times slower than serial with
  the Numba calculator. Set `estimator.settings['workers'] = -1` to use a pool
  for the NumPy calculator and large crops; on macOS and Windows a script that
  does so must be protected by `if __name__ == '__main__':`.
* **`Optimizer` defaults.** The default list of fixed parameters is now every
  model property that is not a property of the particle, plus `k_p`. With an
  aberrated model, the aberration coefficients and center used to be free
  parameters unless you fixed them; they are now fixed. The `robust` argument
  now defaults to `None` and changes the settings only when you pass it. A
  `settings` dictionary that you supply is kept (it used to be overwritten with
  the method and loss of non-robust fitting) and is copied.
* **`Mask` selects an exact number of pixels** (`round(fraction * n)` of the
  `n` pixels that are not excluded) from its own random generator, instead of
  accepting each pixel with probability `fraction`. `fraction` therefore means
  a fraction of the usable pixels, the number of fitted pixels no longer varies
  from draw to draw, and the mask no longer reads or changes NumPy's global
  random state. A new `seed` (last field) makes the selection reproducible.
  `RadialMask` uses the same generator and seed; its pixels are still chosen
  independently, so its count varies.

### Added

* **GPU calculator for Apple silicon.** `TorchLorenzMie` runs a Metal kernel
  through PyTorch's MPS back end. It computes in single precision (hologram
  error about 1e-5 for a 0.5 um sphere, up to about 3e-3 for a 3 um sphere),
  forms the hologram of a single particle inside the kernel, and applies the
  aberration phase of `Aberrated` inside the kernel. On an Apple M1 Pro, a fit
  is two to three times faster than with Numba above about 25,000 pixels and
  about as fast for a few thousand.
* `TorchLorenzMie` accepts `device='cuda'`, `'xpu'` or `'mps'`, and
  `get_device()` chooses among them. The Triton kernels no longer depend on
  CUDA-specific functions, which should let them run on AMD (ROCm) and Intel
  GPUs. The `torch` extra installs Triton on Linux only.
* `Mask(seed=...)`, and `DEEstimator` passes its own seed to its pixel mask, so
  a seed makes the whole estimate reproducible.
* `jaxLorenzMie` includes the aberrations of `Aberrated(jaxLorenzMie)` in its
  hologram and in its analytical Jacobian.
* The example images `crop.png`, `image0010.png` and `image0400.png` are
  installed with the package.
* A `CHANGELOG.md`, `make dev`, and a "Development environment" section in
  `CLAUDE.md`.

### Changed

* **`numbaLorenzMie` is 3.5 to 4 times faster on one thread and up to 20 times
  faster with threads** (a pointwise partial-wave kernel). Threads are used by
  default above 2,048 points (`parallel = True`); see the class documentation
  for when to switch them off. A full fit of one particle is 10 to 17 times
  faster. The calculator can be called from several Python threads at once, and a
  forked worker process (the default start method of `multiprocessing` on Linux
  before Python 3.14) evaluates on one thread, which avoids a hang in OpenMP
  threading layers.
* The NumPy calculator computes direction cosines from ratios instead of
  trigonometric functions, as the other calculators do, and handles a pixel on
  the particle's axis explicitly.
* The JAX tests are left out of the default test run (`pytest
  tests/test_jaxlorenzmie.py` runs them).
* The documentation and all six notebooks are rewritten for the current API
  and the notebooks are re-executed. The Frame example in the quickstart runs
  in about 1 second instead of 7.

### Fixed

* `Aberrated(jaxLorenzMie)` silently ignored every aberration for a single
  sphere, and its analytical Jacobian was that of the unaberrated model.
* `example_hologram()` failed in an installed copy of the package (also in
  2.0.1), because the images were read from the source tree's `docs/`
  directory. That broke every `.example()` method.
* The Read the Docs build, which had failed since the torch backend was added:
  `torchLorenzMie` now imports with `torch` mocked. A new CI job builds the
  documentation the way Read the Docs does.

### Known limitations

* **The CUDA, ROCm and Intel GPU paths of `TorchLorenzMie` have not been run on
  hardware.** Only the Metal path (Apple silicon) has been tested. The Triton
  kernel was checked against a NumPy stand-in, which verifies its mathematics but
  not its compilation, and `TorchLorenzMieBatch` is CUDA-only and was not
  changed. Treat these paths as experimental, and please report results
  from NVIDIA, AMD and Intel GPUs.
* The torch calculators work in single precision.
* `DEEstimator`'s global search can settle on an unphysical solution for hard
  features (for example `n_p` near 2.7). Prior bounds on the search and a
  robust loss, as in the single-frame tutorial, help.

## [2.0.1] and earlier

See the [GitHub releases](https://github.com/davidgrier/pylorenzmie/releases).
