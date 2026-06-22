from pylorenzmie.lib import LMObject
from pylorenzmie.lib.lmtypes import Image, Properties, Result
from pylorenzmie.analysis.Hologram import Hologram
from pylorenzmie.analysis.Mask import Mask
from pylorenzmie.theory import LorenzMie
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares
from scipy.linalg import svd
import pandas as pd


class Optimizer(LMObject):
    '''Fit a generative light-scattering model to holographic data.

    Wraps :func:`scipy.optimize.least_squares`.  The ``method`` class
    attribute must appear as a substring of the model's ``method`` string
    for the pairing to be valid (e.g. ``'numpy'`` matches both
    ``'numpy'`` and ``'cupy numpy'``).

    Parameters
    ----------
    model : LorenzMie, optional
        Generative scattering model.  Default: ``LorenzMie()``.
    mask : Mask or None, optional
        Pixel-selection mask applied to the hologram before fitting.
        ``None`` (default) uses all pixels via
        :attr:`~Hologram.flat_data` and
        :attr:`~Hologram.flat_coordinates`.
    robust : bool, optional
        Use robust Cauchy loss instead of standard least squares.
        Default: ``False``.
    fixed : list[str], optional
        Names of model properties held constant during fitting.
        Default: ``['noise', 'numerical_aperture']``.
    settings : dict, optional
        Keyword arguments forwarded to ``scipy.optimize.least_squares``.
        Defaults to sensible values for holographic particle fitting.
    **kwargs
        Forwarded to ``LorenzMie()`` when ``model`` is not supplied.

    Attributes
    ----------
    result : pandas.Series or None
        Fitted values and uncertainties after :meth:`optimize` has run;
        ``None`` before the first call.
    metadata : pandas.Series
        Fixed model properties and optimizer settings.
    '''

    method: str = 'numpy'

    def __init__(self,
                 model: LorenzMie | None = None,
                 mask: Mask | None = None,
                 robust: bool = False,
                 fixed: list[str] | None = None,
                 settings: Properties | None = None,
                 **kwargs) -> None:
        self.model = model or LorenzMie(**kwargs)
        if self.method not in self.model.method:
            raise TypeError('Model not compatible with Optimizer')
        self.mask = mask
        self.settings = settings
        self.fixed = fixed if fixed is not None else ['noise', 'numerical_aperture']
        self.robust = robust
        self._data: Image | None = None
        self._result = None

    @property
    def properties(self) -> Properties:
        '''Optimizer settings merged with model properties.'''
        properties = dict(settings=self.settings, fixed=self.fixed)
        properties.update(self.model.properties)
        return properties

    @properties.setter
    def properties(self, properties: Properties) -> None:
        self.model.properties = properties
        for name, value in properties.items():
            if hasattr(self, name):
                setattr(self, name, value)

    @property
    def settings(self) -> Properties:
        '''Settings forwarded to ``scipy.optimize.least_squares``.

        Notes
        -----
        ``method``: ``'lm'`` (Levenberg-Marquardt, default), ``'trf'``,
        or ``'dogbox'``. Only ``'lm'`` supports ``loss='linear'``.

        ``loss``: ``'linear'`` (standard LS), ``'cauchy'``, ``'huber'``,
        ``'soft_l1'``, or ``'arctan'``.

        ``x_scale``: ``'jac'`` for dynamic rescaling, or an explicit
        array of scales matched to the free parameters.
        '''
        return self._settings

    @settings.setter
    def settings(self, settings: Properties | None) -> None:
        if settings is None:
            settings = {'method': 'lm',
                        'ftol': 1e-4,
                        'xtol': 1e-6,
                        'gtol': 1e-6,
                        'loss': 'linear',
                        'max_nfev': 2000,
                        'diff_step': 1e-5,
                        'x_scale': 'jac'}
        self._settings = settings

    @property
    def robust(self) -> bool:
        '''True if using robust (non-linear loss) optimization.'''
        return self.settings['loss'] != 'linear'

    @robust.setter
    def robust(self, robust: bool) -> None:
        if robust:
            self.settings['method'] = 'trf'
            self.settings['loss'] = 'cauchy'
        else:
            self.settings['method'] = 'lm'
            self.settings['loss'] = 'linear'

    @property
    def fraction(self) -> float:
        '''Fraction of pixels used for fitting.

        Setting this creates a :class:`Mask` if one does not exist.
        ``1.0`` (all pixels) when no mask is set.
        '''
        return 1.0 if self.mask is None else self.mask.fraction

    @fraction.setter
    def fraction(self, fraction: float) -> None:
        if self.mask is None:
            self.mask = Mask()
        self.mask.fraction = fraction

    @property
    def fixed(self) -> list[str]:
        '''Model properties held constant during fitting.'''
        return self._fixed

    @fixed.setter
    def fixed(self, fixed: list[str]) -> None:
        self._fixed = list(fixed)
        self._variables = [p for p in self.model.properties
                           if p not in fixed]

    @property
    def variables(self) -> list[str]:
        '''Model properties that will be optimized.'''
        return self._variables

    @variables.setter
    def variables(self, variables: list[str]) -> None:
        self._variables = list(variables)
        self._fixed = [p for p in self.model.properties
                       if p not in variables]

    @property
    def result(self) -> pd.Series | None:
        '''Fitted values, uncertainties, and statistics.

        Returns ``None`` before :meth:`optimize` has been called.
        Each fitted parameter ``p`` appears alongside ``d``+``p`` (its
        uncertainty).  Also includes ``success``, ``npix``, and
        ``redchi``.
        '''
        if self._result is None:
            return None
        a = self.variables
        b = ['d' + v for v in a]
        keys = list(sum(zip(a, b), ()))
        keys.extend('success npix redchi'.split())
        values = list(self._result.x)
        npix = self._data.size
        redchi, uncertainties = self._statistics()
        values = list(sum(zip(values, uncertainties), ()))
        values.extend([self._result.success, npix, redchi])
        return pd.Series(dict(zip(keys, values)))

    @property
    def metadata(self) -> pd.Series:
        '''Fixed model properties and optimizer settings.'''
        metadata = {key: self.model.properties[key] for key in self.fixed
                    if key in self.model.properties}
        metadata.update(self.settings)
        return pd.Series(metadata)

    def optimize(self, hologram: Hologram) -> Result:
        '''Fit model to hologram.

        Parameters
        ----------
        hologram : Hologram
            Normalized hologram to fit.  When :attr:`mask` is ``None``
            all pixels are used; otherwise the mask subsamples the
            hologram before fitting.

        Returns
        -------
        result : pandas.Series
            Fitted values, uncertainties, and goodness-of-fit statistics.
        '''
        if self.mask is None:
            self._data = hologram.flat_data
            self.model.coordinates = hologram.flat_coordinates
        else:
            self._data, self.model.coordinates = self.mask.apply(hologram)
        p0 = np.array([self.model.properties[p] for p in self.variables])
        self._result = least_squares(self._residuals, p0, **self.settings)
        return self.result

    def report(self) -> str:
        '''Format fitting results as a human-readable string.

        Returns
        -------
        str
            One line per fitted parameter showing value ± uncertainty,
            followed by pixel count and reduced chi-squared.
        '''
        result = self.result
        if result is None:
            raise RuntimeError('optimize() must be called before report()')
        units = {'x_p': 'pixels', 'y_p': 'pixels', 'z_p': 'pixels',
                 'a_p': 'μm'}
        fmt = {'x_p': '.2f', 'y_p': '.2f', 'z_p': '.2f',
               'a_p': '.3f'}
        lines = []
        for p in self.variables:
            val = result[p]
            err = result['d' + p]
            f = fmt.get(p, '.4f')
            suffix = f' {units[p]}' if p in units else ''
            lines.append(f'{p} = {val:{f}} ± {err:{f}}{suffix}')
        lines += [f'npixels = {result.npix:.0f}',
                  f'χ² = {result.redchi:.2f}']
        return '\n'.join(lines)

    def _residuals(self, values: NDArray[float]) -> Image:
        self.model.properties = dict(zip(self.variables, values))
        noise = self.model.instrument.noise
        return (self.model.hologram() - self._data) / noise

    def _statistics(self) -> tuple[float, NDArray[float]]:
        '''Reduced chi-squared and parameter uncertainties.

        Uncertainties are square roots of the diagonal of the covariance
        matrix, estimated from the Jacobian via Moore-Penrose SVD with
        small singular values discarded.
        '''
        res = self._result
        ndeg = self._data.size - res.x.size
        redchi = 2. * res.cost / ndeg
        _, s, VT = svd(res.jac, full_matrices=False)
        threshold = np.finfo(float).eps * max(res.jac.shape) * s[0]
        s = s[s > threshold]
        VT = VT[:s.size]
        pcov = np.dot(VT.T / s**2, VT)
        uncertainty = np.sqrt(redchi * np.diag(pcov))
        return redchi, uncertainty

    @classmethod
    def example(cls,
                verbose: bool = False,
                **kwargs) -> None:  # pragma: no cover
        from time import perf_counter

        shape = (201, 201)
        model = LorenzMie()
        model.coordinates = model.meshgrid(shape)
        model.particle.a_p = 0.75
        model.particle.n_p = 1.42
        model.particle.r_p = [100., 100., 225.]
        print(cls.__name__ + ' example:')
        print('* Ground truth:')
        print(model.particle)
        noise = model.instrument.noise * np.random.normal(size=shape)
        data = model.hologram().reshape(shape) + noise
        hologram = Hologram(data)
        fixed = 'wavelength magnification numerical_aperture n_m noise k_p'.split()
        a = cls(model=model, fixed=fixed, **kwargs)
        settings = a.settings
        settings['method'] = 'trf'
        settings['loss'] = 'cauchy'
        settings['ftol'] = 1e-3
        settings['xtol'] = None
        settings['gtol'] = None
        if verbose:
            settings['verbose'] = 2
        start = perf_counter()
        a.optimize(hologram)
        print(f'Time to optimize: {perf_counter()-start:.1e} s')
        print('* Fitting results:')
        print(a.report())


if __name__ == '__main__':  # pragma: no cover
    Optimizer.example()
