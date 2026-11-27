'''Least-squares fitting with the residual on an NVIDIA GPU.

:class:`cupyOptimizer` is an :class:`~pylorenzmie.analysis.Optimizer.Optimizer`
for the :class:`~pylorenzmie.theory.cupyLorenzMie.cupyLorenzMie` calculator.
The hologram and the residual are computed on the GPU, and only the residual
vector is copied to the processor for the solver.  It needs the ``cupy``
package.
'''

from typing import Any
import cupy as cp

from pylorenzmie.analysis import Optimizer
from pylorenzmie.lib.lmtypes import Image


class cupyOptimizer(Optimizer):
    '''GPU-accelerated optimizer using CuPy.

    Subclass of :class:`~pylorenzmie.analysis.Optimizer.Optimizer` that keeps
    hologram computation and residual evaluation on the GPU.  The pixels are
    copied to the GPU when a fit starts, and only the residual vector comes
    back to the CPU for the solver.

    Inherits from :class:`~pylorenzmie.analysis.Optimizer`.

    Parameters
    ----------
    double_precision : bool, optional
        If True (default), use double-precision arithmetic on the GPU.
    *args, **kwargs
        Passed to :class:`~pylorenzmie.analysis.Optimizer.Optimizer`.

    Notes
    -----
    The ``method`` attribute is ``'cupy'``, which is a substring of
    ``cupyLorenzMie.method = 'cupy numpy'``, satisfying the optimizer
    compatibility check.
    '''

    method: str = 'cupy'

    def __init__(self, *args: Any,
                 double_precision: bool = True,
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.model.double_precision = double_precision

    @property
    def data(self) -> Image | None:
        '''Pixels being fitted; setting them copies them to the GPU.'''
        return self._data

    @data.setter
    def data(self, data: Image | None) -> None:
        if data is None:
            self._data = self.devicedata = None
        else:
            self._load(data)

    def _load(self, pixels: Image) -> None:
        super()._load(pixels)
        self.devicedata = cp.asarray(pixels, dtype=self.model.dtype)

    def _difference(self) -> Image:
        noise = self.model.instrument.noise
        hologram = self.model.hologram(device=True)
        return ((hologram - self.devicedata) / noise).get()


if __name__ == '__main__':  # pragma: no cover
    print('SINGLE PRECISION')
    cupyOptimizer.example(double_precision=False)
    print('\nDOUBLE PRECISION')
    cupyOptimizer.example()
