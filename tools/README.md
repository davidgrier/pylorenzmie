# tools

Scripts that make files shipped with the package.  They are run by the
maintainer, not imported.

- `train_mlp_estimator.py` trains the network of
  `pylorenzmie.analysis.MLPEstimator` and writes
  `analysis/mlp_estimator.joblib`.  It caches its synthetic training set
  in `tools/mlp_training_data.npz`, which git ignores.
- `make_splash.py` renders `lmtool/splash.png`, the hologram on the
  splash screen of `lmtool`.

Anything that is only a scratchpad belongs in `devel/`, which git ignores.
