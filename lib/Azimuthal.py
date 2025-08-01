import numpy as np
from numpy.typing import NDArray
from functools import wraps


Data = NDArray[float]
Center = NDArray[int | float]
Radii = NDArray[int]
Average = NDArray[float]


def azimuthaloperator(func):
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


def docstring(purpose: str) -> str:
    parameters = '''
    Parameters
    ----------
    data: numpy.ndarray
        Two-dimensional data set
    center: Optional[Tuple(float, float)]
        (x, y) center of azimuthal average
        Default: center of data

    Returns
    -------'''

    def _doc(func):
        outcome = func.__doc__
        func.__doc__ = f'{purpose}\n{parameters}{outcome}'
        return func
    return _doc


@azimuthaloperator
@docstring('Azimuthal average')
def avg(d: Data, r: Radii) -> Average:
    '''
    avg: ndarray
        Average value of data as a function of distance from center
    '''
    nr = np.bincount(r)
    return np.bincount(r, d) / nr


@azimuthaloperator
@docstring('Azimuthal standard deviation')
def std(d: Data, r: Radii) -> tuple[Average, Average]:
    '''
    avg, std: tuple of numpy.ndarray
        Azimuthal average and
        azimuthal standard deviation
        as functions of distance from center
    '''
    nr = np.bincount(r)
    avg = np.bincount(r, d) / nr
    std = np.sqrt(np.bincount(r, (d - avg[r])**2) / nr)
    return avg, std


@azimuthaloperator
@docstring('Azimuthal median')
def med(d: Data, r: Radii) -> Average:
    '''
    med: numpy.ndarray
        Median value as a function of distance from center
    '''
    nmax = r.max() + 1
    med = [np.median(d[np.where(r == n)]) for n in np.arange(nmax)]
    return np.array(med)


@azimuthaloperator
@docstring('Azimuthal median absolute deviation')
def mad(d: Data, r: Radii) -> tuple[Average, Average]:
    '''
    med, mad: tuple of numpy.ndarray
        Azimuthal median and
        azimuthal median absolute deviation
        as functions of distance from center
    '''
    nmax = r.max() + 1
    med = [np.median(d[np.where(r == n)]) for n in np.arange(nmax)]
    dev = [np.median(np.abs(d[np.where(r == n)] - med[n]))
           for n in np.arange(nmax)]
    return np.array(med), np.array(dev)
