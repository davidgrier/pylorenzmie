import numpy as np

def coordinates(shape, corner=None, flatten=True, dtype=float):
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
    (ny, nx) = shape
    if corner is None:
        (left, top) = (0, 0)
    else:
        (left, top) = corner
    x = np.arange(left, nx + left, dtype=dtype)
    y = np.arange(top, ny + top, dtype=dtype)
    xy = np.array(np.meshgrid(x, y))
    if flatten:
        return xy.reshape((2, -1))
    return xy
