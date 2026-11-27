# analysis

The pipeline that finds particles in a hologram and fits them: localize,
estimate, optimize.

- `Hologram.py` holds an image with its coordinates and its saturated
  pixels, and `Mask.py` and `RadialMask.py` choose the pixels that take part
  in a fit.
- `Localizer.py` finds the features in a frame.
- `Estimator.py`, `DEEstimator.py`, `RadialEstimator.py`,
  `MLPEstimator.py` and `PairEstimator.py` estimate starting values
  (`BaseEstimator.py` is their base class).
- `Optimizer.py` (and `cupyOptimizer.py`) refine them with the least
  squares of scipy, Levenberg-Marquardt by default, or with `broyden.py`,
  which updates the Jacobian instead of recomputing it.
- `Feature.py` bundles one particle, `Frame.py` a whole image, and
  `Trajectory.py` links frames in time.  `pair_grouping.py` groups
  features that overlap, when a `Frame` is asked for pairs.
- `mlp_estimator.npz` is the trained network of `MLPEstimator`; retrain
  it with `tools/train_mlp_estimator.py`.

Update this file when the role or contents of the directory change.
