from pylorenzmie.analysis import Optimizer
from pylorenzmie.theory.cupyLorenzMie import cupyLorenzMie
import cupy as cp


class cupyOptimizer(Optimizer):
    '''GPU-accelerated optimizer using CuPy.

    Subclass of :class:`Optimizer` that keeps hologram computation and
    residual evaluation on the GPU.  Only the final residual vector is
    transferred to the CPU for scipy's Levenberg-Marquardt solver.

    Inherits from :class:`~pylorenzmie.analysis.Optimizer`.

    Parameters
    ----------
    double_precision : bool, optional
        If True (default), use double-precision arithmetic on the GPU.
    *args, **kwargs
        Passed to :class:`Optimizer`.

    Notes
    -----
    The ``method`` attribute is ``'cupy'``, which is a substring of
    ``cupyLorenzMie.method = 'cupy numpy'``, satisfying the optimizer
    compatibility check.
    '''

    method: str = 'cupy'

    def __init__(self, *args,
                 double_precision: bool = True,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model.double_precision = double_precision

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data) -> None:
        self._data = data
        self.devicedata = (cp.asarray(data, dtype=self.model.dtype)
                           if data is not None else None)

    def _residuals(self, values: list[float]) -> cupyLorenzMie.Image:
        '''Compute residuals on the GPU; return numpy array to scipy.'''
        self.model.properties = dict(zip(self.variables, values))
        noise = self.model.instrument.noise
        residuals = (self.model.hologram(device=True) - self.devicedata) / noise
        return residuals.get()


if __name__ == '__main__':  # pragma: no cover
    print('SINGLE PRECISION')
    cupyOptimizer.example(double_precision=False)
    print('\nDOUBLE PRECISION')
    cupyOptimizer.example()
