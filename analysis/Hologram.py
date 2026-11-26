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
    saturated : numpy.ndarray of bool, optional
        Pixels at which the camera saturated, shape ``(ny, nx)``.  The
        normalized ``data`` cannot show this, so it comes from the raw
        image (see
        :meth:`~pylorenzmie.utilities.normalizer.Normalizer.saturated`).
        Default: ``None``, no pixel is known to be saturated.

    Raises
    ------
    ValueError
        If ``saturated`` does not have the shape of ``data``.

    Notes
    -----
    Pixel coordinates are generated automatically from ``data.shape``
    and ``corner`` via :func:`~pylorenzmie.lib.meshgrid.meshgrid` and stored as
    a ``(2, ny, nx)`` array.  Two flat views are provided for use by
    scattering models and numerical solvers:

    - :attr:`flat_data` — ``data.ravel()``, shape ``(npts,)``
    - :attr:`flat_coordinates` — ``coordinates.reshape(2, -1)``,
      shape ``(2, npts)``

    Both are numpy views (no copy, O(1)).

    Coordinate-aware cropping::

        crop = hologram[y0:y1, x0:x1]

    returns a new :class:`Hologram` whose ``corner`` is updated so
    that its coordinates are consistent with those of the parent frame,
    with the same part of ``saturated``.
    '''

    data: Image
    corner: tuple[float, float] = (0., 0.)
    saturated: NDArray[np.bool_] | None = None

    def __post_init__(self) -> None:
        self._coordinates: NDArray[np.floating] | None = None
        if self.data is not None:
            self._check_saturated()

    def _check_saturated(self) -> None:
        '''Raise a ValueError if ``saturated`` does not fit ``data``.'''
        if (self.saturated is not None
                and np.shape(self.saturated) != self.shape):
            raise ValueError(f'saturated has shape '
                             f'{np.shape(self.saturated)}, not {self.shape}')

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
    def coordinates(self) -> NDArray[np.floating]:
        '''Pixel coordinates, shape ``(2, ny, nx)``.

        Made from the shape of :attr:`data` and :attr:`corner` when first
        needed.
        '''
        if self._coordinates is None:
            self._coordinates = meshgrid(self.shape,
                                         corner=self.corner,
                                         flatten=False)
        return self._coordinates

    @property
    def excluded(self) -> NDArray[np.bool_]:
        '''Pixels that a fit must not use, shape ``(ny, nx)``.

        The saturated pixels and those whose value is NaN or infinite.
        '''
        self._check_saturated()
        excluded = ~np.isfinite(self.data)
        if self.saturated is not None:
            excluded |= np.asarray(self.saturated, dtype=bool)
        return excluded

    @property
    def flat_data(self) -> Image:
        '''Pixel values as a 1-D view, shape ``(npts,)``.'''
        return self.data.ravel()

    @property
    def flat_coordinates(self) -> NDArray[np.floating]:
        '''Pixel coordinates as a 2-D view, shape ``(2, npts)``.'''
        return self.coordinates.reshape(2, -1)

    @classmethod
    def from_coordinates(cls,
                         data: Image,
                         coordinates: NDArray[np.floating],
                         saturated: NDArray[np.bool_] | None = None
                         ) -> 'Hologram':
        '''Create a Hologram from pixel coordinates that are already known.

        Nothing is computed, so a crop of a frame (or a region that a
        program has cut out of one) keeps the coordinates of the frame.

        Parameters
        ----------
        data : numpy.ndarray
            Image, shape ``(ny, nx)``.
        coordinates : numpy.ndarray
            Pixel coordinates of ``data``, shape ``(2, ny, nx)``.  The
            corner is read from the first pixel.
        saturated : numpy.ndarray of bool, optional
            Saturated pixels of ``data``, shape ``(ny, nx)``.
            Default: ``None``.

        Returns
        -------
        Hologram
            A hologram that shares ``data`` and ``coordinates``.

        Raises
        ------
        ValueError
            If the shape of ``coordinates`` is not ``(2, ny, nx)``, or
            that of ``saturated`` is not ``(ny, nx)``.
        '''
        if coordinates.shape != (2, *data.shape):
            raise ValueError(f'coordinates have shape {coordinates.shape}, '
                             f'not {(2, *data.shape)}')
        corner = (float(coordinates[0, 0, 0]), float(coordinates[1, 0, 0]))
        h = cls(data, corner, saturated)
        h._coordinates = coordinates
        return h

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
            Cropped hologram with coordinates inherited from the parent.
        '''
        sy, sx = key
        self._check_saturated()
        saturated = (None if self.saturated is None
                     else self.saturated[key])
        return Hologram.from_coordinates(self.data[key],
                                         self.coordinates[:, sy, sx],
                                         saturated)

    @classmethod
    def example(cls) -> None:  # pragma: no cover
        '''Run a demonstration of the class, for use as a script.'''
        from pylorenzmie.utilities import example_hologram
        hologram = example_hologram()
        print(f'Hologram: shape={hologram.shape}, corner={hologram.corner}')
        print(f'coordinates: {hologram.coordinates.shape}')
        crop = hologram[50:150, 50:150]
        print(f'Crop: shape={crop.shape}, corner={crop.corner}')


if __name__ == '__main__':  # pragma: no cover
    Hologram.example()
