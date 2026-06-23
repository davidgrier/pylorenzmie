'''Train and save the MLPEstimator weights.

Generates synthetic azimuthal profiles using the 1-D radial trick:
the model is evaluated at coordinates (r, 0) for r = 0..N_FEATURES-1,
which equals the azimuthal average for a rotationally symmetric scatterer
but requires only N_FEATURES model evaluations instead of a full 2-D grid.

Three improvements over the naive uniform-sampling baseline:

1. Log-uniform prior for z_p and a_p: gives equal representation to each
   decade instead of concentrating training signal at large values.  The
   profile is sensitive to z_p and a_p on a multiplicative scale, so this
   matches the geometry of the parameter space.

2. Narrower, physically realistic bounds: avoids the extreme (large a_p,
   high n_p) combinations that overflow the Mie series (~14% of samples
   with wide uniform bounds vs. <1% here).

3. Log+StandardScaler target transformer: the MLP predicts
   (log z_p, log a_p, n_p), each standardized to zero mean / unit variance.
   This compresses the 60:1 dynamic range of z_p and 20:1 range of a_p
   into a well-conditioned output space.  sklearn's TransformedTargetRegressor
   handles the inverse transform at prediction time, so MLPEstimator.estimate()
   needs no changes.

Default instrument: 447 nm laser, 0.048 um/px, water (n_m=1.340).
To retrain for a different instrument, edit INSTRUMENT below and re-run.

Output: ../analysis/mlp_estimator.joblib

Pass --regen to force regeneration of training data even if the cache exists.
'''

import argparse
from pathlib import Path
import time

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
import joblib

from pylorenzmie.analysis.MLPEstimator import _log_targets, _exp_targets
from pylorenzmie.theory import LorenzMie

# ── instrument ────────────────────────────────────────────────────────────────

INSTRUMENT = dict(wavelength=0.447, magnification=0.048, n_m=1.340)

# ── training bounds ───────────────────────────────────────────────────────────
# Narrower than DEEstimator to avoid overflow and focus on realistic particles.
# z_p and a_p are sampled log-uniformly (see generate_data).

BOUNDS = dict(
    z_p=(20.,  500.),   # pixels at default magnification
    a_p=(0.2,   4.0),   # μm  (log-uniform; covers 0.2–4 μm evenly per decade)
    n_p=(1.3,   2.5),   # linear-uniform
)

# ── training parameters ───────────────────────────────────────────────────────

N_TRAIN    = 200_000
N_FEATURES = 100        # radial pixels; must match MLPEstimator.n_features
R_MIN      = 30         # minimum radial profile length during training
SEED       = 42

# ── MLP architecture ──────────────────────────────────────────────────────────

HIDDEN = (256, 128, 64)

# ── paths ─────────────────────────────────────────────────────────────────────

_DEVEL  = Path(__file__).parent
OUTPUT  = _DEVEL.parent / 'analysis' / 'mlp_estimator.joblib'
_CACHE  = _DEVEL / 'mlp_training_data.npz'


def _make_model() -> LorenzMie:
    model = LorenzMie()
    model.instrument.wavelength    = INSTRUMENT['wavelength']
    model.instrument.magnification = INSTRUMENT['magnification']
    model.instrument.n_m           = INSTRUMENT['n_m']
    radii = np.arange(N_FEATURES, dtype=float)
    model.coordinates = np.vstack([radii, np.zeros(N_FEATURES)])
    model.particle.x_p = 0.
    model.particle.y_p = 0.
    return model


def generate_data(model: LorenzMie,
                  rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    '''Return cleaned (X, y): X.shape=(n_good, N_FEATURES), y.shape=(n_good, 3).

    z_p and a_p are drawn log-uniformly; n_p is drawn uniformly.

    Each profile is randomly truncated to R ∈ [R_MIN, N_FEATURES] pixels and
    zero-padded beyond R.  This simulates variable crop sizes at inference
    time and eliminates the train/inference distribution mismatch that occurs
    when small crops are padded with a sentinel after the real data ends.
    At inference, MLPEstimator.estimate() also uses 0-padding, so the
    network has seen the same pattern during training.
    '''
    log_z_lo, log_z_hi = np.log(BOUNDS['z_p'])
    log_a_lo, log_a_hi = np.log(BOUNDS['a_p'])

    z_vals = np.exp(rng.uniform(log_z_lo, log_z_hi, N_TRAIN))
    a_vals = np.exp(rng.uniform(log_a_lo, log_a_hi, N_TRAIN))
    n_vals = rng.uniform(*BOUNDS['n_p'], N_TRAIN)
    r_cuts = rng.integers(R_MIN, N_FEATURES + 1, N_TRAIN)  # truncation radii

    X = np.zeros((N_TRAIN, N_FEATURES))   # 0.0 sentinel for padding
    y = np.column_stack([z_vals, a_vals, n_vals])

    t0 = time.perf_counter()
    with np.errstate(over='ignore', invalid='ignore'):
        for i in range(N_TRAIN):
            model.particle.z_p = z_vals[i]
            model.particle.a_p = a_vals[i]
            model.particle.n_p = n_vals[i]
            R = r_cuts[i]
            X[i, :R] = model.hologram()[:R]   # zeros beyond R (already set)
            if i % 20_000 == 0 and i > 0:
                elapsed = time.perf_counter() - t0
                rate = i / elapsed
                print(f'  {i:>7}/{N_TRAIN}  '
                      f'{elapsed:.0f}s elapsed  '
                      f'{(N_TRAIN - i) / rate:.0f}s remaining')

    print(f'  {N_TRAIN}/{N_TRAIN}  {time.perf_counter() - t0:.1f}s total')

    finite = np.isfinite(X).all(axis=1)
    sane   = (np.abs(X) < 1e4).all(axis=1)
    good   = finite & sane
    n_bad  = N_TRAIN - good.sum()
    if n_bad:
        print(f'  Dropped {n_bad} bad rows ({100 * n_bad / N_TRAIN:.1f}%)')
    return X[good], y[good]


def load_or_generate(regen: bool = False) -> tuple[np.ndarray, np.ndarray]:
    if not regen and _CACHE.exists():
        print(f'Loading cached data from {_CACHE} ...')
        data = np.load(_CACHE)
        return data['X'], data['y']

    print(f'Generating {N_TRAIN:,} synthetic profiles ...')
    rng   = np.random.default_rng(SEED)
    model = _make_model()
    X, y  = generate_data(model, rng)
    np.savez_compressed(_CACHE, X=X, y=y)
    print(f'  Cached → {_CACHE}')
    return X, y


def build_pipeline() -> Pipeline:
    mlp = MLPRegressor(
        hidden_layer_sizes=HIDDEN,
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=SEED,
        verbose=False,
    )
    # Target transformer: log(z_p), log(a_p), n_p → then StandardScaler.
    # TransformedTargetRegressor calls inverse_transform at predict time, so
    # pipeline.predict() returns [z_p, a_p, n_p] in the original scale.
    log_then_scale = Pipeline([
        ('log',   FunctionTransformer(func=_log_targets,
                                      inverse_func=_exp_targets,
                                      validate=False)),
        ('scale', StandardScaler()),
    ])
    return Pipeline([
        ('scaler',    StandardScaler()),
        ('regressor', TransformedTargetRegressor(
            regressor=mlp,
            transformer=log_then_scale,
        )),
    ])


def evaluate(pipe: Pipeline, X: np.ndarray, y: np.ndarray) -> None:
    n_val  = min(10_000, len(X) // 10)
    y_pred = pipe.predict(X[-n_val:])
    y_true = y[-n_val:]
    for j, (label, log_scale) in enumerate([
            ('z_p (px)', True), ('a_p (μm)', True), ('n_p     ', False)]):
        err = np.abs(y_pred[:, j] - y_true[:, j])
        if log_scale:
            rel = np.mean(np.abs(np.log(y_pred[:, j]) - np.log(y_true[:, j])))
            print(f'  {label}  MAE = {np.mean(err):.3f}  '
                  f'(log-MAE = {rel:.3f}, ≈{100*(np.exp(rel)-1):.1f}% typical)')
        else:
            rel = np.mean(err) / np.mean(y_true[:, j])
            print(f'  {label}  MAE = {np.mean(err):.3f}  '
                  f'({100 * rel:.1f}% relative)')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--regen', action='store_true',
                        help='Force regeneration of training data')
    args = parser.parse_args()

    X, y = load_or_generate(regen=args.regen)

    n_val   = len(X) // 10
    X_train, y_train = X[:-n_val], y[:-n_val]

    print(f'\nTraining MLP{HIDDEN} on {len(X_train):,} samples ...')
    pipe = build_pipeline()
    t0   = time.perf_counter()
    pipe.fit(X_train, y_train)
    print(f'  done in {time.perf_counter() - t0:.1f}s')

    print('\nValidation error (held-out 10%):')
    evaluate(pipe, X, y)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, OUTPUT)
    print(f'\nSaved → {OUTPUT}')


if __name__ == '__main__':  # pragma: no cover
    main()
