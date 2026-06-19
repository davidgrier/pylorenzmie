'''Pixel coordinate grid for holographic microscopy images.'''

import numpy as np
from numpy.typing import NDArray


Coordinates = NDArray[float]


def meshgrid(shape: tuple[int, int],
             corner: tuple[float, float] = (0., 0.),
             flatten: bool = True,
             dtype: type = float) -> Coordinates:
    '''Pixel coordinate grid for holographic microscopy images.

    Parameters
    ----------
    shape : tuple[int, int]
        (ny, nx) dimensions of the grid.
    corner : tuple[float, float]
        (left, top) origin of the coordinate system in pixels.
        Default: (0., 0.).
    flatten : bool
        If True (default), return shape (2, ny*nx).
        If False, return shape (2, ny, nx).
    dtype : type
        Numeric type for the coordinate arrays.
        Default: float.

    Returns
    -------
    xy : numpy.ndarray
        Coordinate grid.
    '''
    ny, nx = shape
    left, top = corner
    x = np.arange(left, left + nx, dtype=dtype)
    y = np.arange(top, top + ny, dtype=dtype)
    xy = np.array(np.meshgrid(x, y))
    return xy.reshape((2, -1)) if flatten else xy
