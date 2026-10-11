# TODO

## normalization

- [ ] **Temporal background estimation**: connect `VMedian` to `Normalizer` as a
      `'running'` method. For video data, `VMedian` already accumulates the
      per-pixel temporal median across frames; when particles diffuse, each pixel
      alternates between seeing a particle and seeing clear background, so the
      temporal median converges to the background. Expose this as
      `Normalizer(method='running')` that wraps a `VMedian` instance, updates it
      on each `__call__`, and uses the current running estimate as the reference.
      This is the most reliable single-image-source background method available
      and requires no dedicated reference frame.

- [ ] **Faster single-image background estimation**: replace or supplement the
      `'filter'` method (51-px median filter; slow on large images) with a large
      Gaussian blur or a low-degree 2D polynomial fit. Both are significantly
      faster, handle illumination gradients robustly, and do not leak fringe
      structure into the background estimate when fringe spacing is coarse.
      Gaussian blur: `scipy.ndimage.gaussian_filter(image, sigma=25)`.
      Polynomial fit: `numpy.polynomial.polynomial.polyfit2d` on a downsampled
      version of the image, then evaluate on the full grid.

## estimation

- [x] **MLP estimator**: `analysis/MLPEstimator.py` implemented; pre-trained
      weights ship as `analysis/mlp_estimator.joblib` (bundled via
      `package-data`). Retrain with `devel/train_mlp_estimator.py`.
      `scikit-learn` and `joblib` added as core dependencies.

## lmtool

- [ ] Localizer integration: detect candidate particles in the image; let the
      user toggle through candidates, running Estimator and Optimizer on each
- [ ] Estimation feedback: show before/after parameter values in the status bar
      so the user can judge whether the estimate is reasonable before optimizing
- [ ] Interactive residuals: residual panel currently only updates after a fit;
      update it live as the user adjusts parameters
- [ ] GPU kernels: expose `cupyLorenzMie` and `torchLorenzMie` (CUDA, Metal)
      as options in the GUI
- [ ] Layout: the NA widget was removed with `numerical_aperture`, which left an
      empty cell at row 1, column 0 of `ALMWidget.ui`; rearrange the grid
- [ ] Gamma correction support
- [ ] TOML config file for persistent user preferences

## theory

- [ ] `Cluster` and `Dimer`: review and bring up to production quality
- [x] `torchLorenzMie`: merged. Runs a Metal kernel on Apple GPUs (MPS) and the
      Triton kernel on CUDA, ROCm and Intel (XPU) GPUs; single precision.
      On an M1 Pro it matches Numba for fits of a few thousand pixels and is 2-3
      times faster above about 25,000.
- [ ] `torchLorenzMie`: run `pytest tests/test_torchlorenzmie.py` on CUDA, ROCm and
      Intel hardware (only Metal has been run; the Triton kernel and its
      move to `_tritonkernel.py` are untested), and give `TorchLorenzMieBatch`
      the same treatment
- [x] `torchLorenzMie`: the Metal kernel applies the aberration phase of
      `Aberrated` itself (aberrated fits: 3 times faster than Numba at 90,000
      pixels)
- [ ] `torchLorenzMie`: speed up the Triton kernel as the Metal one was
      (precomputed partial-wave constants, displacement formed in the kernel,
      intensity formed in the kernel, aberration phase in the kernel)
- [ ] `Distorted`: the exact geometric-flow counterpart of `Aberrated` (patch
      on hold; see the aberration manuscript)
- [ ] Piston (constant relative phase) and explicit saturation handling in
      `Feature`/`Aberrated` (patch on hold: the piston needs a physical
      explanation before it is introduced)
- [ ] `LayeredSphere`: core-shell particle class with appropriate Mie coefficients
      a_n/b_n for layered geometries. Relevant for coated beads, lipid vesicles,
      and protein-coated colloids. HoloPy and PyMieScatt both support this;
      pylorenzmie does not.
- [ ] Near-field E and H maps: expose the full near-field spatial distribution
      around a sphere as explicit output alongside `field()` and `hologram()`.
      Relevant for optical trapping force calculations and near-field imaging.
      miepython v3.1+ provides a validated reference implementation.
- [ ] Differentiable forward model: complete `torchLorenzMie` with autograd-
      compatible Mie coefficients (all ops in PyTorch, no NumPy calls; the Metal
      and Triton kernels are hand-written and not differentiable), enabling
      gradient-based optimizers (Adam, L-BFGS), direct backprop into CNN training
      loops, and physics-informed neural network integration. PyMieDiff
      (APL Photonics 2026) demonstrates the pattern for layered spheres.

## precision

- [ ] **Single-precision pipeline**: investigate whether the full pipeline
      (hologram generation, fitting, estimation) can run in float32 rather than
      float64. Single precision halves memory bandwidth and is natively fast on
      consumer GPUs (and on Apple Silicon). Key questions: does LM convergence
      degrade? Are Mie coefficient accumulation errors acceptable at float32?
      The torch backend already uses float32/complex64 internally; the numpy/cupy
      paths use float64 throughout. A natural starting point is to add a
      `dtype` parameter to `LorenzMie` and benchmark fit quality vs. speed.
      Data so far (torch/Metal against float64 NumPy): hologram error 1e-5 for
      a 0.5 um sphere, 6e-4 at 1.5 um and 3e-3 at 3 um, which is below typical
      camera noise but not for the largest spheres; fits agree with Numba to
      the noise.

## optimizer

- [ ] Backend-native Optimizer: implement an `Optimizer` subclass that works
      natively with CuPy (and Torch, and any future backend) so that residuals
      and Jacobians never leave the GPU. Enable `cupyLorenzMie` to return CuPy
      holograms (`device=True`) for fully on-GPU fitting. Track via the `method`
      string convention: a `'cupy'` Optimizer would accept models whose `method`
      contains `'cupy'`.
- [ ] JAX: `jaxLorenzMie` exists (single spheres, analytical Jacobian from
      `jax.jacfwd`, aberrations included) but is opt-in and its tests are left
      out of the default run: on a CPU it is slower than Numba (a fit
      recompiles whenever the number of partial waves changes) and it handles
      only single spheres. Revisit if a use case that needs autodiff or a
      GPU/TPU appears. The original idea:
      a JAX-based `LorenzMie` subclass would
      provide automatic differentiation (exact Jacobians via `jax.jacfwd`),
      JIT-compiled field computation, and cross-platform GPU support (CUDA,
      ROCm, TPU) without separate backends. Pair with `jaxopt.LevenbergMarquardt`
      for a fully on-device fitting pipeline. Would eventually supersede the
      separate numpy/cupy/torch implementations. Requires refactoring the
      multipole loop to `jax.lax.scan` and adopting a functional (non-mutating)
      coding style.
- [ ] Global optimizer stage: add `method='cmaes'` or `method='basin'` options
      to `Optimizer` (wrapping `scipy.optimize.differential_evolution` or
      `basinhopping`) as a coarse-to-fine strategy before LM refinement. Improves
      convergence for poor initial estimates (weak signal, crowded samples).
      HoloPy ships CMA-ES for the same reason.
- [ ] Bayesian posterior estimation: integrate `emcee` (or `dynesty`) as an
      alternative to least-squares so that `Optimizer` can recover the full
      posterior distribution over particle parameters, not just a point estimate
      with χ²-derived error bars. Full posteriors are qualitatively more useful
      for polydisperse or weakly-constrained fits. HoloPy's `emcee` backend is
      the reference design.
- [ ] Random-subset fitting: benchmark and document the `Mask(fraction=...)` +
      `Optimizer` pipeline as an explicit acceleration strategy. HoloPy reports
      >100× speedup fitting tens of pixels vs. a full frame at acceptable accuracy
      cost. The mechanism already exists in pylorenzmie (`Mask` now selects an
      exact number of pixels and takes a `seed` for reproducible fits); what is
      left is to benchmark and document it, and to expose it as a first-class
      option in `Feature` and `Frame`.

- [ ] Joint fitting of several particles in one image (a shared model and
      one residual vector), as used for the Snyder-Grier holograms in the
      aberration study

## detection

- [ ] `Localizer`: `trackpy` returns many false candidates for weak or noisy
      particles (twelve in a synthetic test image of one particle at z = 600
      pixels); add a mass/contrast threshold or a verification step (for
      example, require a minimum number of ring crossings in the radial profile)
- [x] `Localizer` boxes now enclose `nfringes` fringes (two zero crossings each)
      and lie inside the image; the default is `nfringes=4`

## ml

- [x] MLP estimator: implemented in `analysis/MLPEstimator.py`. Pre-trained
      weights for 447 nm laser / 0.048 μm px / water shipped as
      `analysis/mlp_estimator.joblib`. Retrain via `devel/train_mlp_estimator.py`
      (caches generated data in `devel/mlp_training_data.npz` to avoid
      regeneration on re-runs).
- [ ] CNN characterization (CATCH revival): re-implement the CATCH compact CNN
      (Altman & Grier 2020/2023) in PyTorch. Input: normalized 201×201 hologram
      crop. Output: a_p, n_p, z_p. Train on synthetic holograms from the existing
      forward model. Couple to `Localizer` as a drop-in replacement for
      `Estimator`, and use CNN predictions as initial conditions for LM
      refinement. The original TensorFlow 2.2 / Darknet implementation is defunct;
      a PyTorch rewrite would be maintainable and integrate with `torchLorenzMie`.
- [ ] YOLO-based localizer: replace the `CircleTransform` + trackpy localizer
      with a fine-tuned YOLO detection head trained on synthetic and real
      holograms. CATCH (2021) demonstrated this; a modern YOLOv8/RT-DETR variant
      would be faster and more robust to crowded fields.

## tests

- [x] `Optimizer` defaults: the aberration coefficients of an `Aberrated` model
      stay fixed, and user-supplied `settings` are no longer overwritten
- [ ] `test_sphere.py`: add spot-check tests for specific Mie coefficient values
      against published reference data
- [ ] lmtool logic: headless tests for `crop()`, normalization, estimate/undo
      state, and `_handleROIChanged` (pytest-qt)

## api

- [x] `Hologram` class: implemented in v1.2.0. Pairs normalized image data with
      pixel coordinates; `__getitem__` for coordinate-aware slicing; used
      throughout `Feature`, `Frame`, and `FitWidget`.

## infrastructure

- [x] CI: Python 3.11-3.14 with Numba, plus 3.12 without; test jobs time out
      after 20 minutes; Python 3.10 is no longer supported
- [x] Backend selection: `LorenzMie` is `numbaLorenzMie` or the NumPy class;
      GPU backends and JAX are opt-in through `PYLORENZMIE_BACKEND`
- [ ] CI cannot test GPU code (no CUDA or Metal on hosted runners); the torch
      tests need an occasional run on real hardware
- [ ] Development environment: document creating a venv outside the repository
      (a venv inside the tree gets swept by bulk edits), and add `make dev`
      and `.gitignore` entries for `.venv/`, `venv/` and `.lm/`

- [ ] `_setupTheory` in `LMTool`: fragile `layout.replaceWidget` by widget name;
      consider a cleaner injection mechanism
