import numpy as np
from typing import Tuple, Optional


def coordinates(shape: Tuple[int, int],
                corner: Optional[Tuple[int, int]] = None,
                flatten: bool = True,
                dtype=float) -> np.ndarray:
    '''Return coordinate system for Lorenz-Mie microscopy images

    Parameters
    ----------
    shape : tuple
        (nx, ny) shape of the coordinate system

    Keywords
    --------
    corner : tuple
        (left, top) starting coordinates for x and y, respectively
    flatten : bool
        If False, coordinates shape is (2, nx, ny)
        If True, coordinates are flattened to (2, nx*ny)
        Default: True
    dtype : type
        Data type.
        Default: float

    Returns
    -------
    coordinates : numpy.ndarray
        Coordinate system
    '''
    ny, nx = shape
    left, top = (0, 0) if corner is None else corner
    x = np.arange(left, left + nx, dtype=dtype)
    y = np.arange(top, top + ny, dtype=dtype)
    xy = np.array(np.meshgrid(x, y))
    return xy.reshape((2, -1)) if flatten else xy