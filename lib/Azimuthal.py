'''Azimuthal statistics for 2-D images.'''

import numpy as np
from numpy.typing import NDArray
from functools import wraps, lru_cache
from typing import NamedTuple

__all__ = ['azimuthaloperator', 'avg', 'std', 'med', 'mad']

_Data = NDArray[float]
_Center = tuple[float, float] | NDArray[float]
_Average = NDArray[float]


class _Context(NamedTuple):
    '''Cached radial geometry for a given image shape and center.'''
    r: NDArray[np.int32]        # ravelled integer-radius array
    sort_idx: NDArray[np.intp]  # argsort(r); gives radius-ordered data
    boundaries: NDArray[np.intp]  # searchsorted bin edges into sorted r
    nmax: int                   # number of radius bins


@lru_cache(maxsize=16)
def _build_context(ny: int, nx: int, x_p: float, y_p: float) -> _Context:
    '''Compute and cache radial geometry for (ny, nx) centered at (x_p, y_p).'''
    x = np.arange(nx) - x_p
    y = np.arange(ny) - y_p
    r = np.hypot.outer(y, x).astype(np.int32).ravel()
    nmax = int(r.max()) + 1
    sort_idx = np.argsort(r, kind='stable')
    boundaries = np.searchsorted(r[sort_idx], np.arange(nmax + 1))
    return _Context(r=r, sort_idx=sort_idx, boundaries=boundaries, nmax=nmax)


def azimuthaloperator(func):
    '''Decorator that adds image-to-radii conversion to a radial function.

    A function decorated with ``@azimuthaloperator`` gains the public
    signature ``f(data, center=None)``, where *data* is a 2-D image and
    *center* is an ``(x, y)`` coordinate pair defaulting to the image
    centre.  Internally the decorator builds or retrieves a cached
    :class:`_Context` (integer radii, sort order, bin boundaries) and
    passes the ravelled data and context to the original function.

    The context is keyed by ``(ny, nx, x_p, y_p)`` and cached across
    calls via :func:`functools.lru_cache`, so repeated calls on images
    of the same shape and center pay no recomputation cost.
    '''
    @wraps(func)
    def wrappedoperator(data: _Data, center: _Center | None = None):
        ny, nx = data.shape
        x_p, y_p = (nx / 2., ny / 2.) if center is None else center
        ctx = _build_context(ny, nx, float(x_p), float(y_p))
        return func(data.ravel(), ctx)

    return wrappedoperator


@azimuthaloperator
def avg(d: _Data, ctx: _Context) -> _Average:
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
    nr = np.bincount(ctx.r)
    return np.bincount(ctx.r, d) / nr


@azimuthaloperator
def std(d: _Data, ctx: _Context) -> tuple[_Average, _Average]:
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
    nr = np.bincount(ctx.r)
    a = np.bincount(ctx.r, d) / nr
    s = np.sqrt(np.bincount(ctx.r, (d - a[ctx.r]) ** 2) / nr)
    return a, s


@azimuthaloperator
def med(d: _Data, ctx: _Context) -> _Average:
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
    d_sorted = d[ctx.sort_idx]
    result = np.empty(ctx.nmax)
    for n in range(ctx.nmax):
        s, e = ctx.boundaries[n], ctx.boundaries[n + 1]
        result[n] = np.median(d_sorted[s:e]) if s < e else np.nan
    return result


@azimuthaloperator
def mad(d: _Data, ctx: _Context) -> tuple[_Average, _Average]:
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
    d_sorted = d[ctx.sort_idx]
    m = np.empty(ctx.nmax)
    dev = np.empty(ctx.nmax)
    for n in range(ctx.nmax):
        s, e = ctx.boundaries[n], ctx.boundaries[n + 1]
        if s < e:
            ring = d_sorted[s:e]
            med_n = np.median(ring)
            m[n] = med_n
            dev[n] = np.median(np.abs(ring - med_n))
        else:
            m[n] = dev[n] = np.nan
    return m, dev
