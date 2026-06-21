# TODO

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

- [ ] `Hologram` class: pair normalized image data with pixel coordinates so that
      cropping always yields a consistent (data, coordinates) pair. Implement
      `__getitem__` for coordinate-aware slicing (`hologram[y0:y1, x0:x1]`).
      Would simplify `Feature`, `Frame`, and `FitWidget` by collapsing every
      `(data, coordinates)` argument pair into a single object. Sketch in
      `devel/Hologram.py`; revisit when the public API is stabilized.

## infrastructure

- [ ] `_setupTheory` in `LMTool`: fragile `layout.replaceWidget` by widget name;
      consider a cleaner injection mechanism
