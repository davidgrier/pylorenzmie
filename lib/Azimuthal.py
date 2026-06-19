'''Azimuthal statistics for 2-D images.'''

import numpy as np
from numpy.typing import NDArray
from functools import wraps


Data = NDArray[float]
Center = NDArray[int] | NDArray[float]
Radii = NDArray[int]
Average = NDArray[float]


def azimuthaloperator(func):
    '''Decorator that converts a (d, r) function to a (data, center) function.

    The wrapped function receives the raveled data array and the integer
    radius array computed from the image and center coordinates.
    '''
    @wraps(func)
    def wrappedoperator(data: Data,
                        center: Center | None = None,
                        *args, **kwargs):
        ny, nx = data.shape
        x_p, y_p = (nx/2., ny/2.) if center is None else center
        x = np.arange(nx) - x_p
        y = np.arange(ny) - y_p
        d = data.ravel()
        r = np.hypot.outer(y, x).astype(int).ravel()
        return func(d, r, *args, **kwargs)

    return wrappedoperator


@azimuthaloperator
def avg(d: Data, r: Radii) -> Average:
    '''Azimuthal average of a 2-D image.

    Parameters
    ----------
    data : numpy.ndarray
        Two-dimensional data set.
    center : array-like, optional
        (x, y) center for the azimuthal average.
        Default: center of data.

    Returns
    -------
    avg : numpy.ndarray
        Average value of data as a function of distance from center.
    '''
    nr = np.bincount(r)
    return np.bincount(r, d) / nr


@azimuthaloperator
def std(d: Data, r: Radii) -> tuple[Average, Average]:
    '''Azimuthal standard deviation of a 2-D image.

    Parameters
    ----------
    data : numpy.ndarray
        Two-dimensional data set.
    center : array-like, optional
        (x, y) center for the azimuthal average.
        Default: center of data.

    Returns
    -------
    avg : numpy.ndarray
        Azimuthal average as a function of distance from center.
    std : numpy.ndarray
        Azimuthal standard deviation as a function of distance from center.
    '''
    nr = np.bincount(r)
    a = np.bincount(r, d) / nr
    s = np.sqrt(np.bincount(r, (d - a[r])**2) / nr)
    return a, s


@azimuthaloperator
def med(d: Data, r: Radii) -> Average:
    '''Azimuthal median of a 2-D image.

    Parameters
    ----------
    data : numpy.ndarray
        Two-dimensional data set.
    center : array-like, optional
        (x, y) center for the azimuthal median.
        Default: center of data.

    Returns
    -------
    med : numpy.ndarray
        Median value as a function of distance from center.
    '''
    nmax = r.max() + 1
    return np.array([np.median(d[np.where(r == n)])
                     for n in np.arange(nmax)])


@azimuthaloperator
def mad(d: Data, r: Radii) -> tuple[Average, Average]:
    '''Azimuthal median absolute deviation of a 2-D image.

    Parameters
    ----------
    data : numpy.ndarray
        Two-dimensional data set.
    center : array-like, optional
        (x, y) center for the azimuthal median.
        Default: center of data.

    Returns
    -------
    med : numpy.ndarray
        Azimuthal median as a function of distance from center.
    mad : numpy.ndarray
        Azimuthal median absolute deviation as a function of distance
        from center.
    '''
    nmax = r.max() + 1
    m = [np.median(d[np.where(r == n)]) for n in np.arange(nmax)]
    dev = [np.median(np.abs(d[np.where(r == n)] - m[n]))
           for n in np.arange(nmax)]
    return np.array(m), np.array(dev)
