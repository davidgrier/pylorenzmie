'''Random subsets of pixels for fast fitting.

Fitting all of the pixels of a hologram is slow and unnecessary: a random
subset constrains the parameters nearly as well (Dimiduk and others, 2014).
A :class:`Mask` selects ``round(fraction * n)`` of the ``n`` pixels that
are not excluded, so the number of pixels is the same for every draw, using a
random generator of its own.  A ``seed`` makes the selection reproducible, so
that the same hologram is always fitted with the same pixels.  The pixels in
``exclude`` (saturated, dead, NaN or infinite pixels) are never selected.

:meth:`Mask.apply` applies a mask to a
:class:`~pylorenzmie.analysis.Hologram.Hologram` and returns the selected
data and coordinates in the flat form that the solvers use.
:class:`~pylorenzmie.analysis.RadialMask.RadialMask` weights the selection by
distance from the center of the image.
'''

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from pylorenzmie.lib import LMObject
from pylorenzmie.lib.lmtypes import Properties, Image, Coordinates
from pylorenzmie.analysis.Hologram import Hologram


@dataclass
class Mask(LMObject):
    '''Pixel selection mask for subsampling a hologram during fitting.

    Randomly selects a fraction of pixels for analysis, with optional
    exclusion of saturated, NaN, or infinite pixels.  Regenerates
    automatically whenever :attr:`shape`, :attr:`fraction`,
    :attr:`exclude` or :attr:`seed` is changed.

    Exactly ``round(fraction * n)`` of the ``n`` pixels that are not
    excluded are selected, so the number of fitted pixels is the same for
    every draw (compiled and GPU back ends then do not see a new array
    size on each fit).  The generator is a private
    :class:`numpy.random.Generator`: the mask neither uses nor disturbs
    the global NumPy random state.

    Inherits from :class:`pylorenzmie.lib.LMObject`.

    Parameters
    ----------
    shape : tuple[int, int], optional
        ``(height, width)`` of the image.  Default: ``None``.
    fraction : float, optional
        Fraction of pixels to include.  Default: 0.1.
    exclude : numpy.ndarray of bool, optional
        Boolean array of pixels to force-exclude.  Default: ``None``.
    seed : int, optional
        If given, every draw is reproducible: the mask is a function of
        ``seed``, ``shape``, ``fraction`` and ``exclude`` alone, so the
        same hologram is always fitted with the same pixels.  If ``None``
        (default) each draw is independent and unpredictable.

    Notes
    -----
    Call :meth:`update` to resample the mask without changing any
    parameter.  Subclasses may override ``_select`` to implement
    non-uniform sampling strategies.

    References
    ----------
    1. T. G. Dimiduk, R. W. Perry, J. Fung and V. N. Manoharan,
       "Random-subset fitting of digital holograms for fast
       three-dimensional particle tracking,"
       *Applied Optics* **53**, G177–G183 (2014).
    '''

    shape: tuple[int, int] | None = None
    fraction: float = 0.1
    exclude: NDArray[np.bool_] | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        self._mask: NDArray[np.bool_] = np.empty((0, 0), dtype=bool)
        self.update()

    def __setattr__(self, prop: str, value: object) -> None:
        super().__setattr__(prop, value)
        if (prop in ('shape', 'fraction', 'exclude', 'seed') and
                hasattr(self, '_mask')):
            self.update()

    def __call__(self) -> NDArray[np.bool_]:
        return self._mask

    @LMObject.properties.getter
    def properties(self) -> Properties:
        '''Configuration of the mask (the fraction of pixels to sample).'''
        return dict(fraction=self.fraction)

    def _generator(self) -> np.random.Generator:
        '''Random generator for one draw.

        Seeded afresh for every draw when :attr:`seed` is set, so that the
        draw is reproducible; otherwise a persistent generator that
        starts from operating-system entropy.
        '''
        if self.seed is not None:
            return np.random.default_rng(self.seed)
        if not hasattr(self, '_entropy'):
            self._entropy = np.random.default_rng()
        return self._entropy

    def _select(self) -> None:
        '''Randomly select pixels according to :attr:`fraction`.

        Selects ``round(fraction * n)`` of the ``n`` pixels that are not
        excluded.  Subclasses can override this to implement other
        sampling distributions.
        '''
        if self.shape is None:
            return
        allowed = np.ones(self.shape, dtype=bool)
        if self.exclude is not None and self.exclude.shape == allowed.shape:
            allowed &= ~self.exclude
        candidates = np.flatnonzero(allowed)
        k = int(round(min(max(self.fraction, 0.), 1.) * candidates.size))
        chosen = self._generator().choice(candidates, size=k, replace=False)
        mask = np.zeros(allowed.size, dtype=bool)
        mask[chosen] = True
        self._mask = mask.reshape(self.shape)

    def update(self) -> None:
        '''Regenerate the mask with the current parameters.'''
        if self.shape is None:
            return
        self._select()
        if self.exclude is not None and self.exclude.shape == self._mask.shape:
            self._mask[self.exclude] = False

    def apply(self, hologram: Hologram) -> tuple[Image, Coordinates]:
        '''Apply mask to a hologram, returning flat data and coordinates.

        This is the explicit 2D-to-flat boundary for numerical solvers.

        Parameters
        ----------
        hologram : Hologram
            Normalized hologram to subsample.

        Returns
        -------
        data : numpy.ndarray
            Selected pixel values, shape ``(nselected,)``.
        coordinates : numpy.ndarray
            Selected pixel coordinates, shape ``(2, nselected)``.

        Notes
        -----
        Each call draws a new subsample from ``hologram.shape``, unless
        :attr:`seed` is set, in which case the subsample is always the same.

        Both returned arrays are C-contiguous.  Slicing the coordinate
        grid with a 2-D mask yields a non-contiguous view that the
        NumPy backend tolerates but the CuPy backend reads incorrectly,
        so the coordinates are copied to a contiguous layout here.
        '''
        self.shape = hologram.shape
        m = self._mask
        data = hologram.data[m]
        coordinates = np.ascontiguousarray(hologram.coordinates[:, m])
        return data, coordinates

    @classmethod
    def example(cls) -> None:  # pragma: no cover
        import matplotlib.pyplot as plt

        mask = cls(shape=(201, 201), fraction=0.2)
        print(f'fraction = {np.sum(mask()) / mask().size:.2f}')
        plt.imshow(mask(), cmap='gray')
        plt.show()


if __name__ == '__main__':  # pragma: no cover
    Mask.example()
