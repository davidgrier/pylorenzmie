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
- [ ] `cupyLorenzMie`: review; expose in lmtool
- [ ] `torchLorenzMie`: review or retire if superseded

## tests

- [ ] `test_sphere.py`: add spot-check tests for specific Mie coefficient values
      against published reference data
- [ ] lmtool logic: headless tests for `crop()`, normalization, estimate/undo
      state, and `_handleROIChanged` (pytest-qt)

## utilities

- [ ] Review `utilities/` for production quality (normalizer, geometry, h5video, …)

## infrastructure

- [ ] `_setupTheory` in `LMTool`: fragile `layout.replaceWidget` by widget name;
      consider a cleaner injection mechanism
