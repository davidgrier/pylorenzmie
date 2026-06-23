'''Train and save the MLPEstimator weights.

Generates synthetic azimuthal profiles using the 1-D radial trick:
the model is evaluated at coordinates (r, 0) for r = 0..N_FEATURES-1,
which equals the azimuthal average for a rotationally symmetric scatterer
but requires only N_FEATURES model evaluations instead of a full 2-D grid.

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
from sklearn.preprocessing import StandardScaler
import joblib

from pylorenzmie.theory import LorenzMie

# ── instrument ────────────────────────────────────────────────────────────────

INSTRUMENT = dict(wavelength=0.447, magnification=0.048, n_m=1.340)

# ── search bounds (same as DEEstimator.DEFAULT_BOUNDS) ───────────────────────

BOUNDS = dict(
    z_p=(10.,  600.),   # μm/magnification; matches DEEstimator default
    a_p=(0.25,  10.0),  # μm
    n_p=(1.0,    3.0),
)

# ── training parameters ───────────────────────────────────────────────────────

N_TRAIN    = 200_000
N_FEATURES = 100        # radial pixels; must match MLPEstimator.n_features
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
    '''Return cleaned (X, y): X.shape=(n_good, N_FEATURES), y.shape=(n_good, 3).'''
    z_vals = rng.uniform(*BOUNDS['z_p'], N_TRAIN)
    a_vals = rng.uniform(*BOUNDS['a_p'], N_TRAIN)
    n_vals = rng.uniform(*BOUNDS['n_p'], N_TRAIN)

    X = np.empty((N_TRAIN, N_FEATURES))
    y = np.column_stack([z_vals, a_vals, n_vals])

    t0 = time.perf_counter()
    with np.errstate(over='ignore', invalid='ignore'):
        for i in range(N_TRAIN):
            model.particle.z_p = z_vals[i]
            model.particle.a_p = a_vals[i]
            model.particle.n_p = n_vals[i]
            X[i] = model.hologram()
            if i % 20_000 == 0 and i > 0:
                elapsed = time.perf_counter() - t0
                rate = i / elapsed
                print(f'  {i:>7}/{N_TRAIN}  '
                      f'{elapsed:.0f}s elapsed  '
                      f'{(N_TRAIN - i) / rate:.0f}s remaining')

    print(f'  {N_TRAIN}/{N_TRAIN}  {time.perf_counter() - t0:.1f}s total')

    # hologram values are normalised and O(1); filter rows with NaN, Inf,
    # or extreme values that would overflow StandardScaler's variance computation
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
    return Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', TransformedTargetRegressor(
            regressor=mlp,
            transformer=StandardScaler(),
        )),
    ])


def evaluate(pipe: Pipeline, X: np.ndarray, y: np.ndarray) -> None:
    n_val  = min(10_000, len(X) // 10)
    y_pred = pipe.predict(X[-n_val:])
    y_true = y[-n_val:]
    for j, label in enumerate(['z_p (px)', 'a_p (μm)', 'n_p     ']):
        mae = np.mean(np.abs(y_pred[:, j] - y_true[:, j]))
        rel = mae / np.mean(np.abs(y_true[:, j]))
        print(f'  {label}  MAE = {mae:.3f}  ({100 * rel:.1f}% relative)')


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

    print('\nValidation MAE (held-out 10%):')
    evaluate(pipe, X, y)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, OUTPUT)
    print(f'\nSaved → {OUTPUT}')


if __name__ == '__main__':  # pragma: no cover
    main()
