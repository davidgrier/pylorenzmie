from pylorenzmie.analysis import Optimizer
from pylorenzmie.theory.cupyLorenzMie import cupyLorenzMie as LorenzMie
import cupy as cp
import pandas as pd


class cupyOptimizer(Optimizer):
    '''
    Fit generative light-scattering model to data

    ...

    Inherits
    --------
    pylorenzmie.analysis.Optimizer

    Properties
    ----------
    model : cupyLorenzMie
        Generative model for calculating holograms.
    data : numpy.ndarray
        Target for optimization with model.
    properties : dict
        Dictionary of settings for the optimizer as a whole
    settings : dict
        Dictionary of settings for the optimization method
    fixed : list[str]
        Names of properties of the model that should remain constant
        during fitting
    variables : list[str]
        Names of properties of the model that will be optimized.
        Default: All model.properties that are not fixed
    robust : bool
        If True, use robust optimization (absolute deviations)
        otherwise use least-squares optimization
        Default: False (least-squares)
    result : pandas.Series
        Optimized values of the variables, together with numerical
        uncertainties.
    metadata : pandas.Series
        Fixed properties and settings of the optimization method.

    Methods
    -------
    optimize() : pandas.Series
        Optimizes model parameters to fit the model to the data.
        Returns result.

    report() : str
        Returns formatted string of fitting results.
    '''

    method = 'cupy'

    def __init__(self, *args,
                 double_precision: bool = True,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model.double_precision = double_precision

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data: LorenzMie.Image) -> None:
        self._data = data
        self.devicedata = cp.asarray(data, dtype=self.model.dtype)

    def _residuals(self, values: list[float]) -> LorenzMie.Image:
        self.model.properties = dict(zip(self.variables, values))
        noise = self.model.instrument.noise
        residuals = (self.model.hologram(device=True) - self.data) / noise
        return residuals.get()


if __name__ == '__main__':
    print('SINGLE PRECISION')
    cupyOptimizer.example(double_precision=False)
    print('\nDOUBLE PRECISION')
    cupyOptimizer.example()
