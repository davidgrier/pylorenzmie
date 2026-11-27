'''Fast parameter estimator using a pre-trained radial-profile MLP.'''

from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any, Self

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from pylorenzmie.lib import Azimuthal
from pylorenzmie.lib.lmtypes import Properties
from pylorenzmie.analysis.BaseEstimator import BaseEstimator
from pylorenzmie.analysis.Hologram import Hologram
from pylorenzmie.theory import LorenzMie, Sphere


def _default_weights() -> Path:
    folder = files('pylorenzmie.analysis')
    return Path(str(folder.joinpath('mlp_estimator.npz')))


class MLPNetwork:
    '''A trained multilayer perceptron, evaluated with NumPy.

    The network standardizes its input, passes it through fully connected
    layers with ReLU activations (the last layer is linear), and
    un-standardizes the output.  The output is ``(log z_p, log a_p, n_p)``
    in standardized form, so :meth:`predict` returns ``(z_p, a_p, n_p)``.

    ``tools/train_mlp_estimator.py`` trains the network with scikit-learn
    and saves it in this form, so that using one needs only NumPy and no
    particular version of any other library.

    Parameters
    ----------
    x_mean, x_scale : numpy.ndarray
        Mean and standard deviation of each input feature, shape
        ``(n_features,)``.
    weights : list of numpy.ndarray
        Weight matrix of each layer, shape ``(n_in, n_out)``.
    biases : list of numpy.ndarray
        Bias vector of each layer, shape ``(n_out,)``.
    y_mean, y_scale : numpy.ndarray
        Mean and standard deviation of the three outputs after the
        logarithm of the first two, shape ``(3,)``.
    '''

    def __init__(self,
                 x_mean: NDArray[np.floating],
                 x_scale: NDArray[np.floating],
                 weights: list[NDArray[np.floating]],
                 biases: list[NDArray[np.floating]],
                 y_mean: NDArray[np.floating],
                 y_scale: NDArray[np.floating]) -> None:
        self.x_mean = np.asarray(x_mean, dtype=float)
        self.x_scale = np.asarray(x_scale, dtype=float)
        self.weights = [np.asarray(W, dtype=float) for W in weights]
        self.biases = [np.asarray(b, dtype=float) for b in biases]
        self.y_mean = np.asarray(y_mean, dtype=float)
        self.y_scale = np.asarray(y_scale, dtype=float)

    @property
    def n_features(self) -> int:
        '''Length of the input profile.'''
        return self.weights[0].shape[0]

    @classmethod
    def load(cls, path: Path | str) -> Self:
        '''Read a network from a ``.npz`` file.

        Parameters
        ----------
        path : pathlib.Path or str
            File that :meth:`save` or ``tools/train_mlp_estimator.py`` wrote.

        Returns
        -------
        network : MLPNetwork

        Raises
        ------
        ValueError
            If ``path`` is not a ``.npz`` file (a ``joblib`` file of an
            earlier version cannot be read: train again, or convert it).
        '''
        path = Path(path)
        if path.suffix != '.npz':
            raise ValueError(f'MLPEstimator weights are a .npz file, '
                             f'not {path.name!r}')
        with np.load(path, allow_pickle=False) as data:
            n = sum(name.startswith('W') for name in data.files)
            return cls(data['x_mean'],
                       data['x_scale'],
                       [data[f'W{i}'] for i in range(n)],
                       [data[f'b{i}'] for i in range(n)],
                       data['y_mean'],
                       data['y_scale'])

    def save(self, path: Path | str) -> None:
        '''Write the network to a ``.npz`` file.

        Parameters
        ----------
        path : pathlib.Path or str
            Name of the file.
        '''
        arrays: dict[str, Any] = dict(x_mean=self.x_mean,
                                      x_scale=self.x_scale,
                                      y_mean=self.y_mean,
                                      y_scale=self.y_scale)
        layers = zip(self.weights, self.biases, strict=True)
        for i, (W, b) in enumerate(layers):
            arrays[f'W{i}'] = W
            arrays[f'b{i}'] = b
        np.savez_compressed(path, **arrays)

    def predict(self, features: NDArray[np.floating]) -> NDArray[np.float64]:
        '''Estimate the parameters for radial profiles.

        Parameters
        ----------
        features : numpy.ndarray
            Profiles, shape ``(n_samples, n_features)``.

        Returns
        -------
        parameters : numpy.ndarray
            ``z_p`` [pixels], ``a_p`` [um] and ``n_p``, shape
            ``(n_samples, 3)``.
        '''
        x = (np.asarray(features, dtype=float) - self.x_mean) / self.x_scale
        last = len(self.weights) - 1
        layers = zip(self.weights, self.biases, strict=True)
        for i, (W, b) in enumerate(layers):
            x = x @ W + b
            if i < last:
                x = np.maximum(x, 0.)
        y = x * self.y_scale + self.y_mean
        y[:, :2] = np.exp(y[:, :2])
        return y


@dataclass
class MLPEstimator(BaseEstimator):
    '''Fast particle parameter estimator using a pre-trained MLP.

    Computes the azimuthal average of the hologram, pads or truncates it
    to a fixed length, and runs a single forward pass through a
    multilayer perceptron (:class:`MLPNetwork`, evaluated with NumPy) to
    predict *z_p*, *a_p*, and *n_p*
    in one shot.
    Inference takes < 1 ms regardless of image size.

    The MLP was trained on synthetic radial profiles generated by the
    1-D radial trick: ``LorenzMie`` evaluated at ``(r, 0)`` for
    ``r = 0 .. n_features-1``, which equals the azimuthal average for a
    rotationally symmetric scatterer. Parameters are drawn log-uniformly
    over *z_p* ∈ [20, 500] px, *a_p* ∈ [0.2, 4] μm and uniformly over
    *n_p* ∈ [1.3, 2.5].  Each profile is randomly truncated during
    training so the network handles both large and small crops gracefully.
    Pre-trained weights cover the default instrument (447 nm laser,
    0.048 μm/px, water). Retrain with ``tools/train_mlp_estimator.py``
    (which needs scikit-learn) for other configurations.

    Inherits from :class:`~pylorenzmie.analysis.BaseEstimator.BaseEstimator`.

    Parameters
    ----------
    model : LorenzMie
        Generative scattering model shared with
        :class:`~pylorenzmie.analysis.Optimizer.Optimizer`. Particle parameters
        are updated in-place on each :meth:`estimate` call; instrument
        parameters are read but not modified.
    weights : Path or str, optional
        Path to a ``.npz`` file that holds a :class:`MLPNetwork`.
        Default: the pre-trained weights bundled with the package.
    n_features : int, optional
        Fixed input length passed to the MLP. Must match the value used
        during training. Default: 100.

    Notes
    -----
    ``x_p`` and ``y_p`` are those of
    :meth:`~pylorenzmie.analysis.BaseEstimator.BaseEstimator.find_center`,
    as for the other estimators, and the profile is taken about them.

    The profile from :func:`~pylorenzmie.lib.Azimuthal.avg` is truncated
    to the first *n_features* values. Positions beyond the profile end
    are padded with ``0.0`` — a sentinel that is clearly outside the
    normalised hologram range (which is positive and O(1)).  The training
    data uses the same sentinel after random truncation, so the network
    has seen this pattern and uses the zero tail to infer crop size.

    Predicted values are clipped to the training bounds before being written to
    the model; the :class:`~pylorenzmie.analysis.Optimizer.Optimizer` will
    refine them further. '''

    model: LorenzMie
    weights: Path = field(default_factory=_default_weights)
    n_features: int = 100

    def __post_init__(self) -> None:
        self._network = MLPNetwork.load(self.weights)
        if self._network.n_features != self.n_features:
            raise ValueError(f'the network takes '
                             f'{self._network.n_features} features, '
                             f'not n_features={self.n_features}')

    @BaseEstimator.properties.getter
    def properties(self) -> Properties:
        '''MLPEstimator configuration.'''
        return dict(weights=str(self.weights), n_features=self.n_features)

    def estimate(self, hologram: Hologram) -> pd.Series:
        '''Estimate particle parameters from the azimuthal radial profile.

        Parameters
        ----------
        hologram : Hologram
            Normalised hologram crop to analyse.

        Returns
        -------
        result : pandas.Series
            Estimated particle properties (same keys as
            :attr:`~pylorenzmie.theory.Particle.Particle.properties`).
        '''
        x_p, y_p = self.find_center(hologram)

        cx, cy = hologram.corner
        profile = Azimuthal.avg(hologram.data,
                                center=(x_p - cx, y_p - cy))

        features = np.zeros(self.n_features)   # 0 = sentinel for "no data"
        n = min(len(profile), self.n_features)
        features[:n] = profile[:n]

        z_p, a_p, n_p = self._network.predict(
            features.reshape(1, -1))[0]

        particle = self.model.single(Sphere)
        particle.x_p = x_p
        particle.y_p = y_p
        particle.z_p = float(np.clip(z_p,   10.,   600.))
        particle.a_p = float(np.clip(a_p,   0.1,    10.))
        particle.n_p = float(np.clip(n_p,   1.0,     3.0))

        return pd.Series(particle.properties)

    @classmethod
    def example(cls) -> None:  # pragma: no cover
        '''Run a demonstration of the class, for use as a script.'''
        from time import perf_counter
        from pylorenzmie.utilities import example_hologram

        model = LorenzMie()
        model.instrument.wavelength    = 0.447
        model.instrument.magnification = 0.048
        model.instrument.n_m           = 1.34

        estimator = cls(model=model)

        print(f'{cls.__name__} example')
        start = perf_counter()
        result = estimator.estimate(example_hologram())
        print(f'Time: {perf_counter() - start:.3f} s')
        print(result)


if __name__ == '__main__':  # pragma: no cover
    MLPEstimator.example()
