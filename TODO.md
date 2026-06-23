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

- [ ] **MLP estimator** (see ml section below for full spec): a small
      scikit-learn `MLPRegressor` trained on synthetic azimuthal profiles gives
      sub-millisecond inference at DE-quality accuracy. `RadialEstimator` (DE on
      the 1D radial profile) reduces the per-iteration model cost (~10×) but
      DE still runs many iterations, so wall-clock improvement on real data is
      modest. The MLP is the step that actually breaks the seconds barrier.

## lmtool

- [ ] Localizer integration: detect candidate particles in the image; let the
      user toggle through candidates, running Estimator and Optimizer on each
- [ ] Estimation feedback: show before/after parameter values in the status bar
      so the user can judge whether the estimate is reasonable before optimizing
- [ ] Interactive residuals: residual panel currently only updates after a fit;
      update it live as the user adjusts parameters
- [ ] CUDA-accelerated kernels: expose `cupyLorenzMie` as an option in the GUI
- [ ] Gamma correction support
- [ ] TOML config file for persistent user preferences

## theory

- [ ] `Cluster` and `Dimer`: review and bring up to production quality
- [ ] `cupyLorenzMie`: expose in lmtool
- [ ] `torchLorenzMie`: merge from Improvements branch
- [ ] `LayeredSphere`: core-shell particle class with appropriate Mie coefficients
      a_n/b_n for layered geometries. Relevant for coated beads, lipid vesicles,
      and protein-coated colloids. HoloPy and PyMieScatt both support this;
      pylorenzmie does not.
- [ ] Near-field E and H maps: expose the full near-field spatial distribution
      around a sphere as explicit output alongside `field()` and `hologram()`.
      Relevant for optical trapping force calculations and near-field imaging.
      miepython v3.1+ provides a validated reference implementation.
- [ ] Differentiable forward model: complete `torchLorenzMie` with autograd-
      compatible Mie coefficients (all ops in PyTorch, no NumPy calls), enabling
      gradient-based optimizers (Adam, L-BFGS), direct backprop into CNN training
      loops, and physics-informed neural network integration. PyMieDiff
      (APL Photonics 2026) demonstrates the pattern for layered spheres.

## optimizer

- [ ] Backend-native Optimizer: implement an `Optimizer` subclass that works
      natively with CuPy (and Torch, and any future backend) so that residuals
      and Jacobians never leave the GPU. Enable `cupyLorenzMie` to return CuPy
      holograms (`device=True`) for fully on-GPU fitting. Track via the `method`
      string convention: a `'cupy'` Optimizer would accept models whose `method`
      contains `'cupy'`.
- [ ] JAX end-to-end implementation: a JAX-based `LorenzMie` subclass would
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
      cost. The mechanism already exists in pylorenzmie; expose it as a
      first-class option in `Feature` and `Frame`.

## ml

- [ ] MLP estimator: train a small `MLPRegressor` (scikit-learn) on synthetic
      azimuthal profiles to replace `DEEstimator` as the default fast initializer.
      Input: 100-point radial average `b(r)` from `Azimuthal.avg`. Output: z_p,
      a_p, n_p. Training data: millions of synthetic profiles from `LorenzMie` +
      `Azimuthal.avg`, spanning the full physical bounds used by `DEEstimator`.
      Inference: < 1 ms (vs. ~2 s for DE), matching the original Yevick et al.
      (Opt. Express 22, 26884, 2014) SVM approach but with a single multi-output
      MLP instead of three separate SVMs. Ship pre-trained weights for common
      instrument configurations (447 nm laser, silica/PS in water, 0.048 μm/px).
      Provide a `devel/train_mlp_estimator.py` script for retraining on custom
      ranges. Fall back to `DEEstimator` when the MLP confidence is low or the
      predicted values fall outside the training domain.
      Current stepping-stone: `RadialEstimator` (DE on 1D azimuthal profile)
      reduces per-iteration cost but not wall-clock time dramatically; the MLP
      is the step that makes estimation genuinely fast.
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

- [ ] `test_sphere.py`: add spot-check tests for specific Mie coefficient values
      against published reference data
- [ ] lmtool logic: headless tests for `crop()`, normalization, estimate/undo
      state, and `_handleROIChanged` (pytest-qt)

## api

- [x] `Hologram` class: implemented in v1.2.0. Pairs normalized image data with
      pixel coordinates; `__getitem__` for coordinate-aware slicing; used
      throughout `Feature`, `Frame`, and `FitWidget`.

## infrastructure

- [ ] `_setupTheory` in `LMTool`: fragile `layout.replaceWidget` by widget name;
      consider a cleaner injection mechanism
