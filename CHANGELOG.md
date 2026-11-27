# Changelog

All notable changes to pylorenzmie are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [Unreleased]

### Added

* `Optimizer.settings['f_scale'] = 'auto'`, which robust fits now select
  (see Changed), matches the scale of a robust loss
  to the data: it fits by least squares first, and then fits with the loss
  from that solution, with a scale of 1.5 times the spread of those
  residuals.  A scale of 1 (the noise of the instrument, still the default)
  discounts many good pixels when residuals are several times the noise, as
  for real holograms.  On 1,731 holograms of 3 um silica spheres, with the
  Cauchy loss, the scatter between frames fell from that of a scale of 1 by
  40% in the radius, 46% in the index and 36% in the axial position, and
  from that of least squares by 46%, 24% and 27%, with 20% more
  evaluations of the model than a scale of 1 and about twice those of least
  squares.  In simulations of 900 spheres with known values, the error of the
  radius was 4% to 21% below that of a scale of 1.  The reduced chi-squared
  of such a fit is computed with a scale of 1, so that it keeps its meaning
  and can be compared between fits.
* A second fitting method of `Optimizer`, `settings['method'] = 'broyden'`
  (`analysis.broyden.broyden_least_squares`): Levenberg-Marquardt that
  updates the Jacobian from the steps that it takes (Broyden's update), and
  stops when the predicted fall of chi-squared, in units of the reduced
  chi-squared, is below `chi2tol` (default 1e-4).  In double precision, on
  1,120 benchmark holograms, it ended within a tenth of the uncertainties of
  the best fit in 98% of the fits against 96% for the default method, with 6%
  fewer evaluations of the model and in about the same time; for spheres of
  2.5 to 5 um, 96% against 88%, with 6% more evaluations.  On real holograms
  of 3 um silica spheres, whose reduced chi-squared is about 36, 97% against
  94%, with 17% more evaluations.  A tolerance of 0.01 takes 28% fewer
  evaluations than the default method on the benchmark, but ended within a
  tenth of the uncertainties in only 10% of the fits of the silica spheres,
  because the predicted fall is optimistic when the residuals are large.  It
  does least squares without a robust loss or bounds.  In single precision
  (Torch) it does worse than the default method: 58% of the fits against 67%
  ended within a tenth of the uncertainties of the double-precision fit, with
  28% more evaluations.  The default method is unchanged.
* `LorenzMie.diff_steps` gives the steps, by property and in its units, with
  which `Optimizer` differences a calculator.  It is empty for double
  precision.

### Fixed

* Fits with the Torch calculator, which works in single precision, often ended
  far from the minimum: the finite-difference steps, `1e-5` of each value,
  were too small for it (the derivative with respect to `z_p` was wrong by
  tens of percent or more), and a parameter that started at zero, such as an
  aberration, got a step of `1.5e-8`.  `TorchLorenzMie` now gives steps of
  its own.  On an Apple GPU, the share of simulated fits that ended within a
  tenth of their uncertainties of the double-precision fit rose from 44% to
  95%, with 40% fewer evaluations; for spheres of 2.5 to 5 um, the share
  within one uncertainty rose from 19% to 88%.  Not tested on CUDA, and
  `cupyLorenzMie` in single precision is unchanged.
* `cupyOptimizer` could not fit: `optimize` raised `AttributeError`, because
  the pixels were no longer copied to the GPU (since 1.2.0).  A step outside
  the range that the model computes is now rejected there too, as by
  `Optimizer`.  Tested with a stand-in for CuPy, not on a GPU.
* `Optimizer.optimize` left the model at the last point that it had evaluated,
  which was a step of a finite difference: it now leaves it at the fit.
* The documentation of `lmtool`, and its tooltip, said to install CATCH with
  `pip install CATCH`, which installs an unrelated package from PyPI.  They
  now say to install it from its repository.

### Changed

* **SciPy 1.16 or later is required** (from 1.15).  With 1.15, `max_nfev` of
  the `'lm'` method also counted the evaluations of the finite differences,
  so that a limit allowed several times fewer steps than with later versions.
* Setting `Optimizer.robust` to `True` (the `robust` argument, and
  **Optimization > Robust** of `lmtool`) now selects `f_scale='auto'`, unless
  the settings give an `f_scale`.  **Robust fits differ from those of earlier
  versions**, and take longer, because a least-squares fit comes first.  For
  the earlier behavior set `optimizer.settings['f_scale'] = 1`.  Choosing a
  loss in `settings` alone leaves the scale as it was.
* The default stopping tolerances of `Optimizer`, and of `lmtool`, are
  `ftol=1e-7` and `xtol=1e-8` (from `1e-4` and `1e-6`).  Fits often stopped
  short of the minimum that they were approaching: in a benchmark, about a
  third of the fits of example holograms ended within a tenth of their
  uncertainties of it, and nearly all do now.  On simulated holograms with
  the right model the old values were adequate for spheres below 2.5 um.
  The tolerances do not help a fit that approaches the wrong minimum, as
  fits of large spheres can.  **Fitted values differ slightly from those of
  earlier versions** (the radius of a real feature by up to a few tenths of
  a nanometer, typically).  A fit of one sphere takes about 30% longer, and
  60% longer for the example holograms; a fit of a pair of spheres can take
  several times as long, so `Frame` limits each of the pair fits that it
  tries in place of a single sphere to 100 evaluations (`max_nfev`).
  Real holograms that the model explains less well
  can cost more still: for 3 um silica spheres, with a reduced chi-squared of
  about 36, a fit took about twice the evaluations of the model.  Those fits
  ended much closer to their minimum (94% within a tenth of the
  uncertainties, against 1%), but the scatter of the results between frames
  did not change.  For the earlier behavior set
  `optimizer.settings['ftol'] = 1e-4` and
  `optimizer.settings['xtol'] = 1e-6`.

## [3.12.0] - 2026-11-26

### Added

* `Hologram.saturated` marks the pixels at which the camera saturated, and
  `Hologram.excluded` those that a fit must not use (saturated, NaN or
  infinite).  Crops keep them, `Frame` takes them
  (`frame.analyze(data, saturated)`), and `Normalizer.saturated(raw)` finds
  them in a raw image or its background.  `Normalizer(saturation=...)` and
  `lmtool --saturation` set the level, for cameras of 10 or 12 bits that
  write 16-bit files; by default it is the largest value of the type of the
  image.

### Changed

* `Feature` and `lmtool` no longer take the brightest pixels of a region to be
  saturated.  A good pixel was left out of every fit, and when the image was
  divided by a background image, saturated pixels went into the fit, since
  their normalized values differ.  **A hologram that you normalize yourself
  has no saturated pixels unless you name them**: pass
  `Normalizer.saturated(raw)` if your images can saturate.
  `example_hologram` names those of its images.
* `LMTool.raw` has the type of the file (`uint8` or `uint16`, say) and is no
  longer converted to floating point.  Convert it before arithmetic that can
  leave the range of the type.

### Fixed

* `lmtool` estimates ran on all pixels, saturated ones included.
* A fit in `lmtool` failed if the region had a NaN pixel (where the
  background is zero, say).
* `VMedian` cast every frame to eight bits, so that a 16-bit frame gave a
  wrong median (the median of 1000, 3000 and 2000 was 208).  The median now
  has the type of the frames.  Frames of 8 bits are filtered as before; a
  frame of another type starts the filter over, as one of another shape does.
* After `VMedian.reset`, frames from before the reset went into the next
  median, and `VMedian.get` raised an `AttributeError` for a filter that was
  made with a shape.
* `Frame.analyze()` without data discarded the data of the frame.

## [3.11.1] - 2026-11-25

### Fixed

* `utilities.imread` reduced an image of more than eight bits to its high
  byte, without a warning: a 12-bit camera frame in a 16-bit file (values up
  to 4095) became the levels 0 to 15.  It now returns the values of the file
  (`uint16`; `int32` for wider integers, and `float32` for a floating-point
  image, which was clipped to eight bits), and so do `lmtool` and
  `Normalizer.read`.  For such images, a dark count (`--dark`) and a
  background that is given as a number are in the units of the file, and a
  background image must have the bit depth of the hologram.  Eight-bit
  images are read as before.  The examples in the quickstart use `imread`
  in place of OpenCV, which reduces the depth in the same way.
* `lmtool` offered **Use CATCH** when CATCH was not installed but a directory
  named `CATCH` was on the Python path (run from the directory that holds a
  checkout, say), and the task then failed to import it.

## [3.11.0] - 2026-11-25

### Added

* `LayeredSphere`: a sphere with one concentric shell (`t_s`, `n_s`, `k_s`),
  a `Sphere` whose `a_p`, `n_p` and `k_p` are those of the core.  The
  coefficients come from a recursion of Yang's, in the logarithmic
  derivatives of the Riccati-Bessel functions of the incident and the
  outgoing wave (`theory._mie.layered_ab`, compiled with Numba), which is
  stable for thin shells, hollow cores and large spheres; they agree with an
  independent evaluation of Bohren and Huffman's formulas to 1e-9 or better.
  Any calculator that computes with `particle.ab` takes it (NumPy, Numba,
  Torch, CuPy), and an estimator or optimizer takes it where it takes a
  sphere.  `Optimizer` holds `k_s` fixed by default, as it holds `k_p`.
  A thin shell with an index near that of the core fixes only
  `t_s * (n_s**2 - n_m**2)`: fix `n_s` for such a shell (see the docstring).
  The shell and the core must have a positive radius and a thickness that is
  not negative, and the whole sphere is limited to `MAX_RADIUS`; otherwise
  `ab` raises a `ModelDomainError`.

### Fixed

* `jaxLorenzMie` computed a subclass of `Sphere` as a plain sphere, which
  would have left out the shell of a `LayeredSphere`: its kernel and its
  analytical Jacobian are now those of a `Sphere` only, and a layered sphere
  is computed by the NumPy calculator.

## [3.10.1] - 2026-11-25

### Fixed

* The `Types` CI job failed on the optional import of CATCH, which CI does
  not install; pyright now ignores it.  3.10.0 was tagged but not released
  because of it, so 3.10.1 is the first release with the CATCH option of
  `lmtool`.

## [3.10.0] - 2026-11-25

### Added

* `lmtool`: **Particles → Use CATCH**, enabled when the CATCH package is
  installed, makes Find Particles and Estimate use CATCH's detector and
  network instead of the circle transform and differential evolution.
  Nothing from CATCH or PyTorch is imported until a task needs it
  (`tasks.catch_available()` only looks for the package).
* `Frame(estimator=...)`: a factory (a class, say) that makes the estimator
  of each single-particle feature.  A factory, because a feature sets the
  pixel mask of its estimator.
* `tools/train_mlp_estimator.py --output FILE` saves the network somewhere
  other than the file that ships.  `docs/training.rst` says that pylorenzmie
  does not train a localizer.  The `TODO.md` item for a YOLO localizer is
  replaced by one for CATCH's localizer, which is not a YOLO detector.

### Changed

* `Feature.estimate()` writes the returned estimate onto the particle (the
  properties that the model has), so an estimator that only returns its
  result works in a `Feature` and a `Frame`.  The analytic `Estimator`
  returned a result and left the particle as it was, so that `Frame` did
  nothing with it.  The other estimators are unchanged.
* A fit whose starting values give a model that is not finite (a sphere
  closer to the focal plane than its radius, say) raises a
  `ModelDomainError` that names the values, in place of scipy's
  "Residuals are not finite in the initial point".
* `lmtool`: an unhandled exception in a slot is logged and shown in a
  dialog, and the program goes on.  PyQt5 and PyQt6 abort the process on
  one, which lost the session.
* `lmtool`: saving the parameters reports a file that cannot be written in
  the status bar, and no longer truncates the file when the parameters
  cannot be serialized.  The name of a results file that the program makes
  is `results_YYYY-MM-DD_HH-MM-SS.h5`, which sorts by date (it was
  `results_MM_DD_YYYY-...`), in `lmtool.LMTool.RESULTS_DIRECTORY`.
* `lmtool`: the window keeps the background and the dark count in a
  `Background` class that does not need Qt, the hologram in a
  `LoadedImage`, and the names of the tasks in a `Task` enum with one table
  of messages.  The panel of parameters goes into a container that the
  layout provides, in place of a widget found by its name, and a control is
  found by name (`LMWidget.control(name)`) and not with `hasattr`.
* `LorenzMie.warpable` says whether a calculator computes the field from
  displacements, so that `Distorted` can warp it.  The CuPy, JAX and Torch
  calculators set it to `False`; `Distorted` no longer reads the name of the
  method.
* `Hologram.from_coordinates(data, coordinates)` replaces the private
  `Hologram._from_slice`, and raises a `ValueError` for coordinates whose
  shape is not `(2, ny, nx)`.

### Fixed

* `lmtool`: undoing an estimate, or any change of `x_p` that moves the
  region of interest, no longer sets `y_p` back to its old value (setting
  the region to `x_p` announced its position with the old `y_p`).

## [3.9.0] - 2026-11-22

### Added

* `Feature(estimator=...)` takes any estimator, so that an estimator that
  is not part of the package (CATCH's, for example) can start a fit.  The
  default is as before.  `BaseEstimator` declares `exclude`, the pixels
  that an estimate must not use.
* `Optimizer.bounds` (and a `bounds` argument): limits of the variables,
  `{name: (lower, upper)}`, checked against the properties of the model.
  `optimize` uses the trust-region method when there are limits and the
  method of the settings is `lm`, without changing the settings.
* `pylorenzmie.theory.mie_coefficients(x, m, nmax=None)`, the Mie
  coefficients for a size parameter and a relative index, and
  `ModelDomainError`, are exported from `pylorenzmie.theory`.  Code that
  needs the coefficients does not have to import `theory._mie`.

## [3.8.1] - 2026-11-21

### Fixed

* A fit that stepped to a radius of more than 50 um stopped with a
  `ValueError` instead of going on.  The limit (`Sphere.MAX_RADIUS`, added
  in 3.7.0 so that a runaway fit cannot use all the memory of the computer)
  is reached by a step of the solver, which can be a long way from the
  solution.  `Optimizer` now rejects such a step, as it would a step that
  fits worse, and the solver tries a shorter one.  The error is a
  `ModelDomainError`, a `ValueError`, so that `Optimizer` catches that one
  and no other; a starting point outside the range still raises it.  This
  failed `TestFrameOnAnImage.test_results` in the PySide6 job of CI.

## [3.8.0] - 2026-11-21

### Added

* `pylorenzmie.utilities.imread` reads an image file as an 8-bit gray-scale
  array with Pillow (16-bit images are scaled to eight bits, as OpenCV did).
* The dependencies of the package have minimum versions, and a CI job
  (`Lowest dependencies`) runs the tests on Python 3.11 with the lowest
  version of each that `pyproject.toml` allows.

### Changed

* The network of `MLPEstimator` is stored as `analysis/mlp_estimator.npz`
  (arrays) and evaluated with NumPy (`MLPNetwork`), not stored as a pickled
  scikit-learn pipeline.  The pickle depended on the versions of NumPy and
  scikit-learn that wrote it.  The predictions are unchanged, to the last
  bit.  `scikit-learn` and `joblib` are no longer dependencies of the
  package: `scikit-learn` is the `train` extra, for
  `tools/train_mlp_estimator.py`.  `MLPEstimator` refuses a `.joblib` file;
  train again, which writes a `.npz` file.
* OpenCV is no longer a dependency of the package, of the tests, or of
  `tools/`: images are read and written with Pillow, which is now declared
  (it was already installed with Matplotlib, through `trackpy`).  `dgimage`
  uses OpenCV for a faster `CircleTransform` if it is installed.

## [3.7.0] - 2026-11-15

### Added

* `LorenzMie.single(kind=Particle)` returns the particle of a model that has
  one, typed as `kind` (`model.single(Sphere).a_p = 0.75`), and raises
  `TypeError` if the model scatters from a list of particles or from another
  kind of particle.  A list has no position or properties of its own, so the
  code that reads or sets them (the estimators, `Optimizer`, `Pair`) asks for
  the particle in this way.
* `LorenzMie` declares what its backends and wrappers rely on, so that none of
  them probes for it with `hasattr` or `getattr`, or by matching the name of
  a backend: `parallel` (only the Numba calculator uses it; `Estimator` turns
  it off for its own searches), `parallel_threshold`, `fork_safe`,
  `jac_params`, `piston`, and a `jac()` that raises `NotImplementedError` for
  a calculator without an analytical Jacobian.  `DEEstimator` and the
  estimators that share its search raise `ValueError` if `workers` asks for a
  pool of processes and the calculator cannot be forked (`fork_safe` is
  False, as for the CuPy and JAX calculators); `Feature` used to set one
  worker for those, which it already was.
* The accelerated calculators and the code around them are annotated: the
  JAX functions and the Numba kernels, with aliases for what they pass
  (`Array`, `Scalar`, `Reals`, `Complexes`, `Kernel`), the `*args` and
  `**kwargs` of the constructors that pass them on, and the closures of the
  examples.  Only the pointer arguments of the two Triton kernels are not.
* `Bounds` and `Settings` in `pylorenzmie.lib.lmtypes`, for the bounds of a
  search and the keyword arguments of a solver.
* `tools/` holds the scripts that make files that ship with the package,
  `train_mlp_estimator.py` and `make_splash.py` (formerly in `devel/`), with a
  `README.md`; every top-level directory has a `README.md` now.
* The development checks of QVideo and QHOT: `ruff` (rules and line length in
  `pyproject.toml`) and `pyright` run in CI, and tests check the layout of
  argument lists, that the public API has docstrings, that every module can
  be imported first, that `pyproject.toml` lists every package, and that the
  scripts in `tools/` work and the splash image is what `make_splash.py`
  makes.  A `.pre-commit-config.yaml` runs the quick ones before a commit.
  `ruff` is in the `dev` extra.
* CI checks what a user installs and what a release contains: a job builds the
  wheel, installs it outside the checkout, and reads the weights, images,
  layouts and splash image that it ships; a job runs the suite with JAX,
  which the default run leaves out; `actionlint` checks the workflows; and `publish.yml` releases
  nothing unless the Tests workflow passed on the tagged commit.  Dependabot
  keeps the actions current, and the jobs have read access to the repository
  and nothing more.

* `lmtool` runs on PyQt6, PyQt5 and PySide6, and CI tests all three, and
  macOS.  Install `pylorenzmie[pyqt6]` (the same as `[gui]`), `[pyqt5]` or
  `[pyside6]`; the environment variable `QT_API` chooses among the bindings
  that are installed.  `qtpy` is a new dependency.  Without a binding, the
  `lmtool` command says which extra to install.

### Changed

* The Mie coefficients of a sphere are computed by compiled loops when Numba
  is installed (`theory/_mie.py`): 4 to 9 times faster per call (12 to 20 us
  instead of 53 to 183 us), and about 20% off an estimate and a fit of one
  feature (461 to 366 ms), because the global search calls them for every
  candidate.  Without Numba the same loops run as Python.
* The `z_p` bounds of `DEEstimator`, `RadialEstimator` and `PairEstimator`
  are in pixels again, like `x_p`, `y_p` and the `z_p` of a particle, so that
  a name has one unit wherever it appears (3.5.0 made the bounds micrometers).
  The camera independence that 3.5.0 gave the default stays: `bounds` is
  `None` by default, which searches 1 to 200 um at the magnification of the
  instrument (`default_bounds(magnification)`), and it follows a change of
  the magnification.  `DEFAULT_BOUNDS` and `pixel_bounds` are gone.  Code that
  passes `bounds={'z_p': ...}` in micrometers must give pixels instead, and an
  old range is silently a different height: divide it by the magnification.
  The tutorials' bounds are converted at 0.048 um/pixel.
* `lmtool` imports Qt through `qtpy`, with `Signal`, `Slot` and `from qtpy
  import uic`, and `qtpy` is imported before `pyqtgraph` everywhere, so that
  `QT_API` is honored.  The widgets have plain Python properties in place of
  `pyqtProperty`, and `DoubleSlider` and `DoubleSpinBox` no longer connect to
  the signals of their base classes, which PySide6 does not allow.
  `ProfileWidget.upper` and `ProfileWidget.lower`, the curves of the band,
  are `upperLimit` and `lowerLimit`, because `lower` hid `QWidget.lower`.
  `FitWidget.rect` is private for the same reason.
* `Property` (`pylorenzmie.lib.lmtypes`) is what a `properties` dict holds:
  numbers and strings, `None` for a value that is not known yet, and the lists
  and dicts of a configuration.  `Estimator.estimate` and the other estimators
  are annotated as returning a `pandas.Series`, as does `Optimizer.optimize`,
  and `Frame.results`, `Frame.optimize` and `Frame.analyze` a
  `pandas.DataFrame`.  `Particle.r_p` accepts any sequence of three numbers,
  `Optimizer.fixed` and `Optimizer.variables` any sequence of names.
* Errors are reported where they happen.  Setting `LorenzMie.properties` with
  a name that the model does not have, such as a misspelled `a_pp`, raises
  `ValueError`, and sets nothing; it was ignored, and it replaced any
  attribute that the name matched, a method included.  `LorenzMie.properties`
  and the estimators raise `TypeError` for a model that has a list of particles (an
  `AttributeError` before, if they got that far); `Pair` raises `TypeError`
  if its particles are not spheres; `Optimizer` raises `RuntimeError` if it
  is asked about data that it has not been given; `Estimator` raises
  `ValueError` if no sphere matches the profile (a `TypeError` from the next
  line before); `Feature.mask` raises `RuntimeError` if the optimizer has no
  mask; `Frame` raises `RuntimeError` if a feature was not optimized.
* The coverage threshold is 76%, measured in the GUI job of CI, which runs the
  whole suite with Numba and Qt.
* `project.license` is an SPDX expression, `GPL-3.0-or-later`, in place of the
  table that setuptools stops building in February 2027; building needs
  `setuptools>=77`.
* The nine tests of `tests/slow` are in the default suite, and the directory
  is gone.  Three of them analyze a real image of two spheres, and each took a
  minute: the default `Frame` refits a feature whose fit is poor as a pair,
  with a differential-evolution search for each of eight candidates, and on
  this image no pair is better.  Those tests turn the split off and take two
  seconds; the split has tests of its own.
* `Feature` and `Frame` call `Hologram.__init__` in place of setting its
  attributes themselves.  `Hologram` makes its coordinates when they are first
  needed, so a `Frame` that has no image has none, and a crop takes those of
  its parent.
* A task of `lmtool` is cancelled by a model of a subclass of the calculator
  that checks the flag before each hologram and each Jacobian, in place of
  one whose methods were replaced on the instance.
* Imports within a module are grouped as the standard library, third-party
  libraries, `pylorenzmie`, and siblings.  Argument lists are one line, or
  one argument to a line.

### Fixed

* A fit that wandered off to an absurd radius could use all the memory of the
  computer.  The number of partial waves of the Mie sum grows with the radius,
  and a step of an unbounded fit to a radius of -850 um and an index of
  159,000 asked for 1.9 billion of them, 60 GB for the coefficients alone, and
  the system ended the process (a peak of 100 GB; the fit that did it ran
  for 15 s).  `Sphere` refuses a radius of more than `MAX_RADIUS`, 50 um, and
  a sum of more than `MAX_ORDERS`, 5000 partial waves (a sphere of 50 um and
  an index of 3 needs 2,100, the same in air at 300 nm 3,100, one of 5 um in
  water 114), with a `ValueError` that says which radius and index; so does
  `jaxLorenzMie`.  The same fit now stops after 1.6 s with that error.
* `lmtool` could stop for good, about once in a thousand background tasks.
  `TaskRunner` let go of the worker of a task while its thread was still
  running, so that the event loop of the thread destroyed it, and the
  destructor asked for the GIL while it held the lock that Qt keeps for its
  signals; the main thread, which held the GIL, was waiting for that lock.
  The worker now stays with its thread until the thread is done.  This may
  also be the cause of the hang that the macOS job of CI showed twice, with
  the main thread in `LMTool._onCurrentParticle`.
* The Mie coefficients were inaccurate for large spheres of a high index.
  The downward recurrence for the logarithmic derivative `D1` started at the
  highest order that the sum keeps (`nstop` of Wiscombe and Yang), where the
  error of the start has not died out, and the first coefficients inherited
  it.  Against the Bessel functions of the theory, the first ten
  coefficients were off by 1e-6 for a 0.5 um sphere, 1e-4 for 1.5 um of
  polystyrene in water, 0.7% for 3 um, and 100% for 3 um of an index of 2.
  The recurrence now starts
  `15 + 6 nmax^(1/3)` orders higher (`theory/_mie.start_margin`; Yang and
  Bohren and Huffman use a fixed 15, which leaves 3% at 20 um and an index of
  2), and the error is below 1e-13 for radii up to 50 um and indices up to 3.
  `jaxLorenzMie` had the same start and the same fix; `numbaLorenzMie`,
  `TorchLorenzMie` and `cupyLorenzMie` use the coefficients of `Sphere`.
  The effect on a fit is small.  In fits of simulated holograms (2% noise,
  polystyrene in water) the radius, index and height change by less than
  0.0001 um, 0.0001 and 0.03 pixel for spheres up to 3 um, and by 0.004 um
  (0.07%), 0.002 and 0.3 pixel for 5 um, far below the uncertainties usually
  quoted for holographic characterization, so published results are not
  likely to have been affected significantly.  The cost is about 1 us for each
  evaluation of the coefficients, under 0.5% of the time of an evaluation of
  the model.
* `PairEstimator` searched the height of a pair in the wrong units.  The
  bounds on `z_p` were micrometers, and `DEEstimator` and `RadialEstimator`
  converted them to pixels with the magnification; `PairEstimator` used them
  as they were, so that the search stopped at 200 pixels, about 10 micrometers
  at 0.048 um/pixel, and could not find a pair above it.  A pair at 250 pixels
  was estimated at 200.  The split of a poor fit into a pair, which uses the
  search, found no pair in `image0010.png` for that reason.
* `PairEstimator` did not use its `seed` for the subsample of pixels that its
  search fits (`DEEstimator` does), so a seed did not make its estimate
  reproducible.  A frame that splits a poor fit into a pair gave a different
  fit from run to run, and about one run in ten of the test of the split
  ended with a reduced chi-squared of 3 where it expected less than 2.
* `lmtool` worked on PyQt6 only: on PyQt5, toggling a particle in the table
  of particles raised an error.
* A `ProfileWidget` did not move the particle of its model to the origin: it
  set an attribute on the model, not on the particle.
* The analytical Jacobian of `jaxLorenzMie` is finite at the pixel that lies
  exactly on the axis of the particle, where the gradient of the distance
  was 0/0.  A fit of a particle centered on a pixel failed with `array must
  not contain infs or NaNs`.
* The example of `Hologram` built a `Hologram` from a `Hologram`.
* `torchLorenzMie` named `Particle` in two annotations without importing it,
  and the signature of `OptimizerWidget` named a `QWidget` that it does not
  import.

### Removed

* `compute_field_jit` of `pylorenzmie.theory.numbaLorenzMie`, a wrapper that
  nothing but its test called (its `scratch` argument did nothing).
* `lib/obsolete/` and `theory/obsolete/`, 26 files that nothing imported
  (the history has them), and the untracked scratch directory `devel/` is
  ignored by git.
* The `Makefile`.  The commands that it ran are in `CLAUDE.md` and
  `CONTRIBUTING.md`.

## [3.6.0] - 2026-11-13

### Changed

* `Azimuthal` and `CircleTransform` now come from
  [dgimage](https://github.com/davidgrier/dgimage), the small library of image
  transformations (numpy and scipy only) that pylorenzmie shares with QVideo
  and other projects of the group, so that there is one implementation of
  each.  `pylorenzmie.lib.Azimuthal` and `pylorenzmie.lib.CircleTransform`
  keep their names and their numbers (identical to rounding, tested).
  `dgimage>=0.2.0` is a new dependency.  Two things come with it: the average
  of an image about a center that has not been used takes 9 ms instead of
  88 ms for 1280 x 1024 (the sort by radius, which only `med` and `mad` need,
  is made lazily and with a radix sort), which helps every search that moves a
  center, and `dgimage.circletransform` has options (`dtype=np.float32`,
  `binning`) for a faster transform that the `Localizer` does not use.

* **Coma, astigmatism, field curvature and distortion now use the position of
  each ray in the image plane**, ``(r - r_c) / wavelength``, in
  `Aberrated`, `Distorted`, their kernels (Numba, JAX and Metal) and
  `generators.generator`.  Earlier versions used one constant for the
  whole hologram, the displacement of the particle from the aberration
  center, which evaluated the ray coordinate at the particle but the
  momentum at the image plane.  The aberrations of the optical train act on
  the field in the image plane, so the position of the ray there is the
  right coordinate (Table I of Gronert et al.).  **The numerical values of the
  coefficients of these aberrations change**: they are instrumental constants
  and are much smaller than before (the image spans many wavelengths).
  Defocus and spherical aberration, and the results that use them only, are
  unaffected.  An aberration centered on the particle no longer vanishes.
  `generators.generator` takes the vacuum wavelength in pixels as the new
  argument `scale`, and the Numba warp kernels take `1 / scale` as an extra
  last argument.  A coefficient `c` of the unreleased convention is the
  Seidel coefficient `W_klm / wavelength = l NA^l (H / wavelength)^k c`; see
  the docstring of `Aberrated`.

### Fixed

* The tutorials (`SingleFrame` and `Video_analysis`) gave the `z_p` bounds of
  the estimator in pixels. Since 3.5.0 they are micrometers, so the fits of
  the notebooks were sent thousands of pixels away. The ranges are converted
  at 0.048 um/pixel (`(4.8, 16.8)` and `(2.9, 14.4)` um).

## [3.5.1] - 2026-11-07

### Changed

* The field-dependent aberrations (coma, astigmatism, distortion, and field
  curvature in `Distorted`) now scale with
  `eta = (distance from the aberration center) / (vacuum wavelength)` instead
  of `/ z_p`, so that their coefficients are instrumental constants that do
  not depend on the particle's height or on the refractive index of the
  medium. Fitted coma, astigmatism and distortion coefficients from earlier
  versions are not comparable; defocus, spherical aberration and piston are
  unchanged. The JAX and Metal kernels take the vacuum wavelength in pixels
  as an eighth aberration coefficient.

### Fixed

* The JAX aberration phase has exact gradients at a pixel on the particle's
  axis; the distortion term's derivative there was previously zeroed by a
  guard.

## [3.5.0] - 2026-10-27

### Added

* `lmtool` shows a splash screen, a hologram of a particle rendered with
  `LorenzMie` (`devel/make_splash.py` makes `lmtool/splash.png`), while the
  analysis pipeline loads, which takes seconds and longer on the first run
  after an installation. The command line is in `pylorenzmie.lmtool.arguments`
  and the start-up in `pylorenzmie.lmtool.launch`, so that nothing slow is
  imported before the splash screen appears. `BACKENDS` is defined in
  `pylorenzmie.backends` and is still exported by `pylorenzmie.theory`.
* Two overlapping particles can be fitted together. `Pair`
  (`pylorenzmie.theory`) is a `Cluster` of two spheres with independent
  positions, radii and indices, for particles that do not touch (touching
  ones are a `Dimer`). `PairEstimator` extends `DEEstimator` to it: it takes
  the two centers from the caller (or from the pair) and searches `z_p`, `a_p`
  and `n_p` of both spheres with the bounds of `DEEstimator`.
  `Feature(centers=[(x1, y1), (x2, y2)])` builds a pair feature.
  `pair_grouping.group_overlapping` merges the localized boxes that overlap
  into one box with `n_particles` equal to 2.
* `Frame.split_redchi` (default 2.0) and `Frame.split_suspects()`: a
  single-particle fit whose reduced chi-squared exceeds `split_redchi` is
  refit as a pair, from eight starting arrangements about the single fit, and
  the pair replaces it only if its reduced chi-squared is lower. `None`
  switches the check off.
* `Localizer` takes `percentile` and `separation`, which it passes to
  `trackpy.locate`, as attributes and as overrides in `localize`.
* `Normalizer` takes the dark count (black level) of the camera, `dark`, and
  subtracts it from the image and the reference before it divides. Without it
  the fringes of a hologram have too little contrast by the factor
  `(background - dark) / background`, and fits compensate with a particle that
  scatters less (a smaller radius, a larger index). `lmtool` takes it as
  `--dark` (`-D`) and `LMTool.setDark`, and shows it in the status bar. The
  `Normalizer` notes say how to find it from the noise when there is no dark
  image.

### Changed

* `Frame.detect` groups overlapping boxes with `group_overlapping`: two
  overlapping boxes become one pair feature, and three or more are discarded,
  since a box that holds pixels of a third particle cannot be fitted as a
  single particle or a pair. `Frame.optimize` then calls `split_suspects`, so
  the results of a frame can contain pairs by default; pass
  `split_redchi=None` for the previous behavior.
* `Mask.apply` returns C-contiguous coordinates, because the CuPy backend read
  the non-contiguous slice incorrectly. `Feature` runs the differential
  evolution on one worker with the CuPy and JAX backends, so that it does not
  fork the state of the GPU.
* The `lmtool` command, `python -m pylorenzmie.lmtool` and the entry point
  start through `pylorenzmie.lmtool.launch`. `argument_parser` and `lmtool`
  are no longer in `pylorenzmie.lmtool.LMTool`; reinstall the package
  (`pip install -e .`) to update the `lmtool` script.
* `Estimator` finds the center of the particle and matches the profile to a
  sphere to obtain the radius. The radius came from the Fraunhofer
  approximation and was about 4 um for every sphere in our tests, whatever its
  size. It is now the radius whose Lorenz-Mie profile at `n_p` best matches the
  azimuthal average, searched on a grid of radii and of heights about the
  paraxial estimate, and the height of the match replaces the paraxial one.
  The profile is taken about the point at which the hologram is most nearly
  rotationally symmetric, found within `max_shift` (10 pixels by default; zero
  keeps the center of the crop) of the center of the crop: an error of 10
  pixels in the center made the radius of a micron-sized sphere too small by
  25% to 40% and moved a larger one by 25% in the other direction. `x_p` and
  `y_p` of the result are therefore no longer the center of the crop.
* The `z_p` bounds of `DEEstimator` and `RadialEstimator` are in micrometers,
  like `a_p`, and the search converts them to pixels with the magnification of
  the instrument. A range in pixels covered a different height in the sample
  at each magnification: the old default of 10 to 600 pixels stopped at 72 um
  at 0.120 um/pixel, so a particle at 990 pixels (119 um) was out of reach of
  the estimate in `lmtool`. The default is now 1 to 200 um. Code that passes
  `bounds={'z_p': ...}` in pixels must give micrometers instead.

### Fixed

* `lmtool`: a region of interest that extends past the edge of the image is
  drawn where its pixels are, instead of the clipped pixels being stretched
  over the whole region.

## [3.4.0] - 2026-10-16

`Distorted` models the aberrations as the exact distortions of the operator
formalism, and `lmtool` can use it, and any backend.

### Added

* `pylorenzmie.theory.Distorted.Distorted`, a factory like `Aberrated` whose
  calculator moves the points of the image along the flow of each aberration
  and evaluates the scattered field there, instead of multiplying it by a phase
  factor. It takes the coefficients, units and signs of `Aberrated`, plus
  `fieldcurvature`, and costs 1.3 to 1.8 times the ideal model in a fit with
  one to three aberrations. The flows of different aberrations do not commute,
  so the default `mode='composed'` defines the coefficients together with the
  `order` of the operators (the order of `generators.NAMES`);
  `mode='summed'` follows the flow of the sum of the generators and does not
  depend on the order. The two agree to first order. It works with the NumPy and
  Numba calculators. The Numba warp kernels are cached on disk, and a model
  compiles or loads them when it is made, so that no thread compiles one
  while it is running. `pylorenzmie.theory.DistortedLorenzMie` follows the
  backend like `AberratedLorenzMie`.
* `Normalizer.read(background)`, a normalizer that divides by a number or by an
  image read from a file, and `Normalizer.source`, the name of that file.
* `pylorenzmie.theory.generators`, the generating function of each aberration
  and the Runge-Kutta flow.
* `pylorenzmie.theory.model_class(backend, distorted)` and
  `pylorenzmie.theory.BACKENDS`: the aberrated or distorted model for a named
  backend.
* `lmtool --distorted` (`-d`) and `lmtool --backend` (`-B`; `-b` is still
  `--background`) choose the model and the calculator. A choice that cannot be
  used, such as `-d` with torch, is reported at startup. `ALMWidget.with_model`
  makes the panel class for a model class.

### Fixed

* `lmtool image.png --background file` no longer ends in a traceback when the
  file exists but is not an image.

### Changed

* `Normalizer` no longer guesses. An unknown `method`, a scalar reference that is
  not positive and finite, the `'reference'` method without a reference, and a
  reference image of another shape than the image raise `ValueError`, where they
  used to fall back to the median, or to a warning in the log, and a pixel where
  the background is not positive is NaN, where it used to be left in raw counts.
  `Mask` excludes NaN pixels. Code that relied on the fallbacks must catch the
  error or give a matching reference.
* `lmtool`: the background comes from `Normalizer.read`, for `--background` and
  for **Open Background** alike, so a number, a file name, a missing file and a
  file that is not an image are handled in one place. The status bar shows the
  background in use, and says so when a hologram is divided by the median
  because the background does not fit it or could not be read. The `normalizer`
  argument of `LMTool` is replaced by `background`.
* `lmtool`: the dialog for **Open Hologram** starts at the current file, and the
  dialog for **Open Background** at the last background or, failing that, in
  the folder of the hologram.
* The error for a backend that cannot be loaded names the backend, instead of
  the environment variable `PYLORENZMIE_BACKEND`, which may not have been set.

## [3.3.1] - 2026-10-12

`lmtool` uses Numba, and so does everything else that uses `AberratedLorenzMie`.

### Changed

* `pylorenzmie.theory.AberratedLorenzMie` is built on the calculator that
  `pylorenzmie.theory.LorenzMie` selects (Numba if it works, or the backend named
  by `PYLORENZMIE_BACKEND`), not always on NumPy. `lmtool`, which uses it, was
  running its holograms 11 times, its fits 8 times and its estimates 5 times
  slower than necessary. The results agree to 1e-9. The NumPy class is still
  `pylorenzmie.theory.Aberrated.AberratedLorenzMie`.

## [3.3.0] - 2026-10-12

`lmtool` is rebuilt around a session that holds the parameters of a fit, with
estimates, fits and searches for particles that run in the background and can be
cancelled, and a table of the particles in the image and their results.

### Removed

* `FitWidget.optimize`, `optimizeAsync` and its signals, `FitWidget.model`,
  `FitWidget.saveResult` and the other ways that it saved, and
  `lmtool/LMTool.json`. They were internal to `lmtool`; the replacements are
  below.

### Changed

* `lmtool`: the parameters of a fit live in an `LMSession`, which holds the
  model, the value of every parameter and which parameters are fixed; the
  controls, the profile and the fit display are views of it. One Estimate
  recomputes the display once instead of four times, and one Optimize three
  times instead of nine. A parameter without a control is part of the session,
  fixed.
* `lmtool` looks for particles in the background, and the window stays usable
  while it does; loading another image replaces a search that is running.
  `Localizer.localize` takes `check`, a function that it calls before each
  feature and that can raise to stop it.
* `lmtool` has a Particles tab, a table of the particles that were found or
  added (**Particles -> Add Particle**, Ctrl+Shift+A) with a check box to leave
  one out, the status of each (new, estimated, fitted) and its fit. An estimate
  or a fit is recorded for the particle that the ROI is on, and clicking a row
  restores the particle and its parameters. **Save Result** writes the fits of
  the checked particles, one row each, as HDF5, JSON or CSV, with the metadata
  of the fits; it used to write one particle.
* `lmtool`: an estimate or a fit runs in the background on a `Snapshot`, a
  frozen copy of the parameters, the region and the settings, with a model of its
  own; it shares nothing with the interface thread, and the session takes its
  result when it is done. **Optimization -> Cancel** (Esc) stops the task and
  changes nothing. `FitWidget.optimize`, `optimizeAsync` and the signals of the
  optimization are gone, and so are `LMSession.adopt` and `restore`, which only
  the shared model needed.
* `lmtool`: a `Preview` computes what the model predicts (the hologram of the
  region and the radial profile) with models of its own, from the parameters of
  the session, and only once however many changes came before the event loop ran:
  an Estimate or an Optimize computes one prediction, and five parameters set
  in a row compute one. The display no longer reads the model that a fit is
  changing in another thread, which closes a data race (switching to the Fit tab
  during a fit).
* `lmtool`: the parameters of the panel are described by `ParameterSpec`
  entries in `lmtool/parameters.py` (label, range, step, decimals, unit, start
  value, fixed by default), which replace `LMTool.json`. A test checks that every
  control of every panel has a specification and belongs to its model.

## [3.2.0] - 2026-10-12

`lmtool` finds the particles in an image and estimates its noise, saving results
works again, and `Localizer` can rank and limit its candidates.

### Added

* `lmtool` looks for particles when it loads an image and puts the ROI on the
  first, sized to its fringes. **Particles -> Next Particle** (Ctrl+N) selects
  the others and **Find Particles** (Ctrl+L) searches again.
* `Localizer.localize` takes `maxfeatures`, which keeps the strongest
  features first, and returns each feature's `mass` and the number that
  `trackpy` found in `predictions.attrs['nfound']`. The bounding box of each
  feature costs about 80 ms on a full frame, so a noisy 1280 x 1024 image with
  226 candidates took 20 s; limiting them takes one second.
* `pylorenzmie.utilities.estimate_noise` estimates the standard deviation of
  the pixel noise in an image from a second-difference filter, which ignores the
  fringes of a hologram and a smooth background. It agrees with white Gaussian
  noise to 3%.
* `lmtool` sets the noise of the model with it when an image is loaded, and
  **Optimization -> Estimate Noise** does it for the ROI. The noise used to be
  a fixed 0.05, so the residuals and the reduced chi-squared were scaled by a
  number unrelated to the data.
* `lmtool` keeps the ten strongest candidates, and says in the status bar when
  there are none, only one, or many.
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
