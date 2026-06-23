'''Azimuthal statistics for 2-D images.'''

import numpy as np
from numpy.typing import NDArray
from functools import wraps

__all__ = ['azimuthaloperator', 'avg', 'std', 'med', 'mad']

_Data = NDArray[float]
_Radii = NDArray[np.intp]
_Center = tuple[float, float] | NDArray[float]
_Average = NDArray[float]


def azimuthaloperator(func):
    '''Decorator that adds image-to-radii conversion to a radial function.

    A function decorated with ``@azimuthaloperator`` gains the public
    signature ``f(data, center=None)``, where *data* is a 2-D image and
    *center* is an ``(x, y)`` coordinate pair defaulting to the image
    centre. Internally the decorator computes integer pixel radii and
    passes the raveled data and radius arrays to the original function.
    '''
    @wraps(func)
    def wrappedoperator(data: _Data,
                        center: _Center | None = None):
        ny, nx = data.shape
        x_p, y_p = (nx / 2., ny / 2.) if center is None else center
        x = np.arange(nx) - x_p
        y = np.arange(ny) - y_p
        d = data.ravel()
        r = np.hypot.outer(y, x).astype(int).ravel()
        return func(d, r)

    return wrappedoperator


@azimuthaloperator
def avg(d: _Data, r: _Radii) -> _Average:
    '''Azimuthal average of a 2-D image.

    Parameters
    ----------
    data : numpy.ndarray
        Two-dimensional data set.
    center : tuple of float, optional
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
def std(d: _Data, r: _Radii) -> tuple[_Average, _Average]:
    '''Azimuthal standard deviation of a 2-D image.

    Parameters
    ----------
    data : numpy.ndarray
        Two-dimensional data set.
    center : tuple of float, optional
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
def med(d: _Data, r: _Radii) -> _Average:
    '''Azimuthal median of a 2-D image.

    Parameters
    ----------
    data : numpy.ndarray
        Two-dimensional data set.
    center : tuple of float, optional
        (x, y) center for the azimuthal median.
        Default: center of data.

    Returns
    -------
    med : numpy.ndarray
        Median value as a function of distance from center.
        Radii with no contributing pixels are set to NaN.
    '''
    nmax = r.max() + 1
    result = np.empty(nmax)
    for n in range(nmax):
        mask = r == n
        result[n] = np.median(d[mask]) if mask.any() else np.nan
    return result


@azimuthaloperator
def mad(d: _Data, r: _Radii) -> tuple[_Average, _Average]:
    '''Azimuthal median absolute deviation of a 2-D image.

    Parameters
    ----------
    data : numpy.ndarray
        Two-dimensional data set.
    center : tuple of float, optional
        (x, y) center for the azimuthal median.
        Default: center of data.

    Returns
    -------
    med : numpy.ndarray
        Azimuthal median as a function of distance from center.
        Radii with no contributing pixels are set to NaN.
    mad : numpy.ndarray
        Azimuthal median absolute deviation as a function of distance
        from center.
    '''
    nmax = r.max() + 1
    m = np.empty(nmax)
    dev = np.empty(nmax)
    for n in range(nmax):
        mask = r == n
        if mask.any():
            ring = d[mask]
            m[n] = np.median(ring)
            dev[n] = np.median(np.abs(ring - m[n]))
        else:
            m[n] = dev[n] = np.nan
    return m, dev
