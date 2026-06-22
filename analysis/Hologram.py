'''Normalized hologram image with pixel coordinates.'''

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from pylorenzmie.lib import meshgrid
from pylorenzmie.lib.lmtypes import Image


@dataclass(eq=False)
class Hologram:
    '''Normalized hologram paired with pixel coordinates.

    Parameters
    ----------
    data : numpy.ndarray
        Normalized hologram, shape ``(ny, nx)``.  Pixel values should
        be floating-point intensities normalized so the background
        level is approximately 1 (i.e. ``I(r) / I_0(r)``).
    corner : tuple[float, float], optional
        ``(left, top)`` pixel coordinates of the top-left corner of
        this image within the full camera frame.  Default: ``(0., 0.)``.

    Notes
    -----
    Pixel coordinates are generated automatically from ``data.shape``
    and ``corner`` via :func:`~pylorenzmie.lib.meshgrid` and stored as
    a ``(2, ny, nx)`` array.  Two flat views are provided for use by
    scattering models and numerical solvers:

    - :attr:`flat_data` — ``data.ravel()``, shape ``(npts,)``
    - :attr:`flat_coordinates` — ``coordinates.reshape(2, -1)``,
      shape ``(2, npts)``

    Both are numpy views (no copy, O(1)).

    Coordinate-aware cropping::

        crop = hologram[y0:y1, x0:x1]

    returns a new :class:`Hologram` whose ``corner`` is updated so
    that its coordinates are consistent with those of the parent frame.
    '''

    data: Image
    corner: tuple[float, float] = (0., 0.)

    def __post_init__(self) -> None:
        self._coordinates = meshgrid(self.data.shape,
                                     corner=self.corner,
                                     flatten=False)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Hologram):
            return NotImplemented
        return (np.array_equal(self.data, other.data) and
                self.corner == other.corner)

    @property
    def shape(self) -> tuple[int, int]:
        '''Image dimensions ``(ny, nx)``.'''
        return self.data.shape

    @property
    def coordinates(self) -> NDArray[float]:
        '''Pixel coordinates, shape ``(2, ny, nx)``.'''
        return self._coordinates

    @property
    def flat_data(self) -> NDArray[float]:
        '''Pixel values as a 1-D view, shape ``(npts,)``.'''
        return self.data.ravel()

    @property
    def flat_coordinates(self) -> NDArray[float]:
        '''Pixel coordinates as a 2-D view, shape ``(2, npts)``.'''
        return self._coordinates.reshape(2, -1)

    def __getitem__(self, key: tuple) -> 'Hologram':
        '''Return a coordinate-aware crop.

        Parameters
        ----------
        key : tuple[slice, slice]
            ``(slice_y, slice_x)`` row and column slices, as produced
            by a bounding-box from :class:`~pylorenzmie.analysis.Localizer`.

        Returns
        -------
        Hologram
            Cropped hologram.  The ``corner`` is updated so that its
            coordinates are consistent with the parent frame.
        '''
        sy, sx = key
        x0 = self.corner[0] + (sx.start if sx.start is not None else 0)
        y0 = self.corner[1] + (sy.start if sy.start is not None else 0)
        return Hologram(self.data[key], corner=(x0, y0))

    @classmethod
    def example(cls) -> None:  # pragma: no cover
        from pylorenzmie.utilities import example_hologram
        hologram = cls(example_hologram())
        print(f'Hologram: shape={hologram.shape}, corner={hologram.corner}')
        print(f'coordinates: {hologram.coordinates.shape}')
        crop = hologram[50:150, 50:150]
        print(f'Crop: shape={crop.shape}, corner={crop.corner}')


if __name__ == '__main__':  # pragma: no cover
    Hologram.example()
