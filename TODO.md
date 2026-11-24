# TODO

## normalization

- [ ] **Background estimation from the hologram itself** (the biggest obstacle
      to analysis: features are hard to see and to fit until the background is
      normalized away, and a blank frame or a long sequence is often not
      available). A benchmark of the real blank frame in `docs/tutorials`
      multiplied by simulated Lorenz-Mie spheres showed:
      the blank frame is dominated by a fixed pattern (about 24% rms below 25
      pixels, against 8% pixel noise), so smooth backgrounds (median, Gaussian,
      polynomial) leave 28-50% error; with them `Localizer` finds 110-150 false
      particles and fits lose a factor 10-30 in precision; the `'filter'` method
      (51-pixel median filter) is the worst for size and index, worse than the
      scalar median. The model can remove the particles, which a frame cannot
      do alone. Plan:
      1. an FFT **template-bank detector** (matched filter with model holograms
         over height, size and index): found both test particles at 58 and 15
         sigma within 1 pixel, against a pattern floor of 10 sigma; it feeds
         `Estimator` with centers, which `DEEstimator` currently takes from the
         box;
      2. a **`BackgroundEstimator`** with `update(frame)`: detect, fit, divide
         the particles out, and average recursively with a smooth gain for
         drift. Fit-and-divide alone recovers the background to 3.5% rms (smooth:
         36%); averaged over 8 frames of particles that move only 15-80 pixels
         per frame, 1.2-1.3% (temporal median: 2.0-3.2%), and fits then match
         those with the true background;
      3. `Normalizer(method='model')` that uses it, and a documentation note
         about `'filter'`.
      A frame normalized with a background estimated from itself reproduces the
      model, so the single-frame estimate does not improve that frame's own
      fit. Whitening the fit residuals by the pattern spectrum made the fits
      worse in a first test. Not yet tested: drift, particles that are missed
      or not spheres, a low-rank (PCA) library of instrument backgrounds, a
      joint fit with a smooth gain.
- [ ] **Learned methods for the background problem**: (a) train the CATCH-style
      detector and estimator on simulated holograms multiplied by real blank
      frames, so that the network learns to ignore the pattern; (b) a
      learned prior for the background (PCA first, then a VAE or a Gaussian
      process) from the group's blank frames, inside the fit-and-divide scheme;
      (c) amortized inference of the particle parameters with uncertainties,
      marginalized over real background patches. Keep the physical model for
      the final fit. See the CNN item under `ml`.

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
      weights ship as `analysis/mlp_estimator.npz` (bundled via
      `package-data`) and run in NumPy alone (`MLPNetwork`). Retrain with
      `tools/train_mlp_estimator.py`, which needs the `train` extra
      (`scikit-learn`).

## lmtool

- [x] Localizer integration: the ROI goes to the first candidate when an image
      is loaded, and Particles -> Next Particle toggles through the others
- [ ] Batch mode: run Estimator and Optimizer on every candidate and collect
      the results in one table
- [ ] Estimation feedback: show before/after parameter values in the status bar
      so the user can judge whether the estimate is reasonable before optimizing
- [x] Interactive residuals: the residual panel updates as parameters change,
      and the model is no longer clipped to the range of the data
- [ ] GPU kernels: expose `cupyLorenzMie` and `torchLorenzMie` (CUDA, Metal)
      as options in the GUI
- [x] Layout: noise, piston and spherical fill the second row of the
      instrument controls
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
- [x] `Distorted`: the exact geometric-flow counterpart of `Aberrated`, with a
      fused Numba warp kernel; `lmtool --distorted` and `--backend`
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
- [ ] **Analytic Jacobian for the Numba backend** (measured 2026-11-24 from
      CATCH 3.0.0 on its frozen test set, Linux, TITAN RTX, 8 cores; a
      starting point, not a settled design).  `numbaLorenzMie.jac_params` is
      empty, so `Optimizer` leaves the Jacobian to
      `scipy.optimize.least_squares`, which differences it: about 6 model
      evaluations per iteration for the 5 free parameters.  In CATCH's fits
      (4000 pixels, `loss='both'`, crops of 330-590 pixels) one evaluation
      costs 1.3-4 ms and the number of evaluations is very uneven: median 61
      per fit, but 300-1800 for spheres of 4-5 um (1-7 s per fit).  Those fits
      are most of the cost (spheres of 4 um and more take 58% of the CPU time
      of a 1382-particle run: 1789 CPU-s for 5 starts each).  With
      `jaxLorenzMie`, which has the analytic Jacobian, the same fits needed
      far fewer calls (1409 evaluations became 85 + 76 Jacobian calls; 1806
      became 308 + 301), but JAX on a CPU recompiles per partial-wave count
      and took 117 s against 18 s for 12 fits, so that timing says nothing
      about a native Jacobian.
      * **Tried 2026-11-24 and reverted: a wash as built.**  A
        `numbaLorenzMie.jac()` with a fused kernel (analytic derivatives
        with respect to `x_p`, `y_p`, `z_p`; central differences of the
        Mie coefficients, at a fixed number of orders, for `a_p` and
        `n_p`) matched finite differences to 1e-9 and `jaxLorenzMie.jac()`
        to 1e-7, but did not make fits faster.  On 720 paired fits
        (201x201, 2% noise, polystyrene in water, 0.5-5 um, z = 150-600
        px, 4000 pixels, one thread; starts from the truth displaced by 1%
        and 2% in `z_p`, `a_p`, `n_p`, and from `Estimator`):
        lock-on 45.0% against 44.9% (the same outcome in 717 of 720
        pairs), total time 0.96 of the finite-difference run, the same
        number of iterations.  The Jacobian cost 3.8 to 4.7 hologram
        evaluations (0.5 to 4.5 um), and `lm` with differences spends 5
        per iteration, so the saving is about 5.5 against 6 evaluations
        per iteration.  The 16 fits that needed 300 or more evaluations
        (up to 2715) took 74.7 s against 67.3 s, and none locked on.
        **The JAX comparison above (1409 evaluations became 85 + 76
        Jacobian calls) counts the evaluations of the differences, not
        iterations, so it does not predict a speedup.**  A native Jacobian
        pays only if it costs about 2 evaluations or less (then about 2x);
        the kernel carried 20 complex accumulators per partial wave, so the
        place to start would be a profile of it, or a Jacobian for the
        position only with the coefficients differenced.  It also needs a
        guard: `Aberrated` and `Distorted` inherit `jac_params` from the
        base class, which would be wrong for a Jacobian that leaves out
        the phase mask or the warp (`jaxLorenzMie` applies the phase in
        its kernel).  The failures of fits above 2 um come from the
        starts, not from the derivatives (see the review item on the
        estimation of large spheres).
      * Idea: a Jacobian from the recurrence that gives the field (the
        derivatives with respect to `x_p`, `y_p`, `z_p` are a displacement of
        the field; `a_p` and `n_p` enter through `a_n`, `b_n`), added to
        `numbaLorenzMie` with `jac_params` set, so that `Optimizer` passes it
        to `least_squares` as it does for JAX.
      * **Not automatically lossless.**  In 12 fits, 9 gave the same `a_p` to
        3 decimals with the two Jacobians; in 3 they ended in different
        places of the degenerate valley of large spheres (for one, 5.04 um
        with reduced chi-squared 2.13 by finite differences against 4.86 um
        and 2.26 with JAX).  Stopping rules (`ftol`, `xtol`, `x_scale='jac'`)
        and step paths differ.  Check on many fits: the lock-on rate (radius
        within 5 nm of the truth) and the reduced chi-squared against the
        current `Optimizer` on simulated holograms of 0.5-5 um.  CATCH's
        `python -m CATCH.customization.evaluation.scene_run --set test`
        (frozen set of 400 scenes, 1381 particles) scores it end to end; CATCH
        3.0.0 locks on 67.8% from the best start and 72.8% from all 5.
      * Not worth it: a batched GPU solver on `TorchLorenzMieBatch`.  Its
        kernel makes a hologram of 4000 pixels in 0.15 ms (B = 64 to 256,
        float32) against 0.75 ms for one Numba core, i.e. about 6,500
        holograms per second against 10,000 for 8 cores, and the single-fit
        GPU paths (`cupyOptimizer`) send each residual back to the host for
        scipy.  Re-test if the Triton kernel gets faster (see `theory`).
      * Related, from CATCH: fitting only the best start, or gating the other
        starts on the reduced chi-squared, did not help (the reduced
        chi-squared of a fit that locks on and of one that does not overlap:
        medians 0.8 and 1.5); the extra starts matter above 2 um (4.7 points
        of lock-on, none below 2 um).
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

- [ ] **Revisit `Localizer`.** It finds only one of the particles in a
      PS/silica image (`PS_silica.png`) that has several, and many false
      candidates in a noisy one (`maxfeatures` ranks them and limits the cost;
      it does not decide what is a particle). The circle transform with
      `trackpy.locate` and one `diameter` for all sizes is the likely limit;
      look at multiscale detection, a threshold on `mass`, and verification by
      a radial profile or a template bank (see the background-estimation item).

- [ ] `Localizer`: `trackpy` returns many false candidates for weak or noisy
      particles (twelve in a synthetic test image of one particle at z = 600
      pixels); add a mass/contrast threshold or a verification step (for
      example, require a minimum number of ring crossings in the radial profile)
- [x] `Localizer` boxes now enclose `nfringes` fringes (two zero crossings each)
      and lie inside the image; the default is `nfringes=4`

## ml

- [x] MLP estimator: implemented in `analysis/MLPEstimator.py`. Pre-trained
      weights for 447 nm laser / 0.048 μm px / water shipped as
      `analysis/mlp_estimator.npz`. Retrain via `tools/train_mlp_estimator.py`
      (caches generated data in `tools/mlp_training_data.npz` to avoid
      regeneration on re-runs).
- [ ] CNN characterization (CATCH revival): re-implement the CATCH compact CNN
      (Altman & Grier 2020/2023) in PyTorch. Input: normalized 201×201 hologram
      crop. Output: a_p, n_p, z_p. Train on synthetic holograms from the existing
      forward model. Couple to `Localizer` as a drop-in replacement for
      `Estimator`, and use CNN predictions as initial conditions for LM
      refinement. The original TensorFlow 2.2 / Darknet implementation is defunct;
      a PyTorch rewrite would be maintainable and integrate with `torchLorenzMie`.
- [ ] CATCH localizer: offer CATCH's trained localizer (a CenterNet-style
      detector) as an alternative to the `CircleTransform` + trackpy
      `Localizer`, behind the same `x_p`, `y_p`, `bbox` table and with a lazy
      import so that pylorenzmie does not need PyTorch.  First compare the two
      on the frozen CATCH test set and on recorded frames, and decide
      whether to replace or to offer both.

## tests

- [x] `Optimizer` defaults: the aberration coefficients of an `Aberrated` model
      stay fixed, and user-supplied `settings` are no longer overwritten
- [x] Mie coefficients: `tests/test_mie.py` compares them with the Bessel
      functions of the theory for radii of 0.1 to 30 um and indices of 1 to 3
      (and fails when the start of the recurrence is too low)
- [ ] `test_sphere.py`: spot-check Mie coefficients against published
      reference data (tables of Bohren and Huffman), not only against the
      Bessel functions
- [x] lmtool logic: headless tests of the window, the widgets, the session and
      the tasks (`tests/test_lmtool*.py`, 150 tests, Qt drawing to no display),
      run on PyQt6, PyQt5 and PySide6

## review (2026-11)

Open findings of the review of the code base; the notes behind them are in
`CHANGELOG.md` (what was done) and in the commit messages.

- [ ] **Pair-split rule** (`Frame._split`): a pair is accepted when its
      reduced chi-squared is lower than the single fit's, with no margin.
      After the unit fix both features of `image0010.png` become pairs
      (3.6 to 2.6 and 3.9 to 2.8), which may be overfitting.  Decide a margin
      (or an F-test) and look at how many splits survive on the test images.
- [ ] **Estimation of large spheres** (investigated 2026-11-15; a decision is
      open).  Simulated holograms (201x201, 2% noise, polystyrene or silica in
      water, 0.5-5 um, z = 150-600 px, centered or 3 px off) show that the LM
      fit is not the problem: given a start inside the valley it recovers the
      truth to under 1 nm.  The valley is narrow for large spheres (the cost
      has a local minimum every 15-30 nm in `a_p` at 3 um, and the valley of
      the truth is about 12 nm wide), so the start must be good.  A fit is a
      success if `a_p` is within 5 nm of the truth, which coincides with a
      reduced chi-squared below 1.1.
      * **Width of the basin.**  LM started from the truth, with `z_p`, `a_p`
        and `n_p` all displaced by a relative amount eps (spheres of 2 um and
        more): 88% succeed at 0.5%, 68% at 1%, 42% at 2%, 0% at 5%.  A start
        must be within 1 to 2% in every parameter at once.
      * **Starts, with `n_p` unknown** (180 fits, bounded LM on 4000 pixels):
        `DEEstimator` then LM 48%; the analytic `Estimator` then LM 38%;
        a scan of `n_p` (and `z_p`) from the analytic start 28 to 36%.  Above
        2 um none exceeds about 50%, and at 5 um all are near 20%.  An earlier
        run of the default `DEEstimator` gave 57% (103 of 180); more pixels,
        a larger population or a tighter `tol` help little, and narrow bounds
        help a lot (3 and 5 um: 8 of 8 with `z_p` 150-450 px, `a_p` 0.5-6 um,
        `n_p` 1.3-1.8), which is why the tutorials work.
      * **The analytic `Estimator` needs `n_p`.**  It cannot estimate it (the
        default is 1.5).  An earlier entry gave 96% for it; that test most
        likely told it the true index, which makes it 29 of 45 fits at 2 to
        4 um against 5 of 45 with the default.  On the real holograms of
        `tests/data` and `utilities/data` it ends at reduced chi-squared
        13-26 against 3.6-5.5 for `DEEstimator`.
      * **A fit on a growing radius does not help.**  LM on the pixels within
        25, 40, 60, 85, 120 pixels and then all, with `x_p` and `y_p` held
        fixed on the small disks: 52% from `DEEstimator`, 42% from the
        analytic start, and from the truth displaced by eps it is no better
        than a single fit (85% against 88% at 0.5%, 35% against 42% at 2%).
        A global search on the inner disk first: 44%, at 6 s a fit.
      Open: a start that is within 1 to 2% needs more than these estimators;
      a CATCH-style network would have to reach that accuracy (the
      `MLPEstimator` is at about 9%).  Not yet tried: real data of large
      spheres, a multi-start in `a_p` at the width of the valley, tighter
      default bounds.  `DEEstimator` returned `n_p = 2.65` for one real
      feature (`image0400.png`, reduced chi-squared 9.6), an unphysical
      solution that a bound on `n_p` would reject.
- [ ] The Triton kernel exists twice (`_tritonkernel.py` and
      `torchLorenzMieBatch.py`, about 100 lines); unify with CUDA hardware to
      test it.  The two kernels have no type annotations (`@triton.jit` reads
      them).
- [ ] The macOS GUI job hung once (the stack was in `LMTool._onCurrentParticle`
      during the event loop) and has not since; a stuck test now ends the run
      after five minutes with the stacks of its threads.  Watch the next runs; the `TaskRunner` fix
      (a worker destroyed in its thread while the main thread held the GIL)
      may be the cause.

## api

- [x] `Hologram` class: implemented in v1.2.0. Pairs normalized image data with
      pixel coordinates; `__getitem__` for coordinate-aware slicing; used
      throughout `Feature`, `Frame`, and `FitWidget`.

## infrastructure

- [x] CI: ruff and `actionlint`; pyright on the whole package; the tests on
      Python 3.11-3.14 with Numba and on 3.12 without; the GUI tests on PyQt6,
      PyQt5 and PySide6 and on macOS (coverage in one of them); JAX on a CPU;
      the wheel, installed outside the checkout; the documentation with
      warnings as errors.  A release waits for the
      tests of its commit.  Python 3.10 is no longer supported
- [x] Backend selection: `LorenzMie` is `numbaLorenzMie` or the NumPy class;
      GPU backends and JAX are opt-in through `PYLORENZMIE_BACKEND`
- [ ] CI cannot test GPU code (no CUDA or Metal on hosted runners); the torch
      tests need an occasional run on real hardware
- [x] Development environment: a venv outside the repository is documented in
      `CLAUDE.md` and `CONTRIBUTING.md`, and `.gitignore` has `.venv/`,
      `venv/` and `.lm/`

- [x] `LMTool._setupControls` (was `_setupTheory`): the layout has an empty
      container with a layout of its own, and the panel is added to it, instead
      of replacing a placeholder found by its name
