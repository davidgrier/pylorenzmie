from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from pylorenzmie.lib import LMObject
from pylorenzmie.lib.types import Properties


@dataclass
class Mask(LMObject):
    '''Pixel selection mask for subsampling a hologram during fitting.

    Randomly selects a fraction of pixels for analysis, with optional
    exclusion of saturated, NaN, or infinite pixels.  Regenerates
    automatically whenever :attr:`shape`, :attr:`fraction`, or
    :attr:`exclude` is changed.

    Inherits from :class:`pylorenzmie.lib.LMObject`.

    Parameters
    ----------
    shape : tuple[int, int], optional
        ``(height, width)`` of the image.  Default: ``None``.
    fraction : float, optional
        Fraction of pixels to include.  Default: 0.1.
    exclude : numpy.ndarray of bool, optional
        Boolean array of pixels to force-exclude.  Default: ``None``.

    Notes
    -----
    Call :meth:`update` to resample the mask without changing any
    parameter.  Subclasses may override :meth:`_select` to implement
    non-uniform sampling strategies.
    '''

    shape: tuple[int, int] | None = None
    fraction: float = 0.1
    exclude: NDArray[np.bool_] | None = None

    def __post_init__(self) -> None:
        self._initialized = False
        self._mask: NDArray[np.bool_] = np.empty((0, 0), dtype=bool)
        self.update()
        self._initialized = True

    def __setattr__(self, prop: str, value: object) -> None:
        super().__setattr__(prop, value)
        if prop in ('shape', 'fraction', 'exclude'):
            if getattr(self, '_initialized', False):
                self.update()

    def __call__(self) -> NDArray[np.bool_]:
        return self._mask

    @LMObject.properties.getter
    def properties(self) -> Properties:
        '''Mask configuration: fraction of pixels to sample.'''
        return dict(fraction=self.fraction)

    def _select(self) -> None:
        '''Randomly select pixels according to :attr:`fraction`.

        Subclasses can override this to implement other sampling
        distributions.
        '''
        if self.shape is None:
            return
        self._mask = np.random.rand(*self.shape) < self.fraction

    def update(self) -> None:
        '''Regenerate the mask with the current parameters.'''
        if self.shape is None:
            return
        self._select()
        if self.exclude is not None:
            self._mask[self.exclude] = False

    @classmethod
    def example(cls) -> None:  # pragma: no cover
        import matplotlib.pyplot as plt

        mask = cls(shape=(201, 201), fraction=0.2)
        print(f'fraction = {np.sum(mask()) / mask().size:.2f}')
        plt.imshow(mask(), cmap='gray')
        plt.show()


if __name__ == '__main__':  # pragma: no cover
    Mask.example()
