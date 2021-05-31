from .circletransform import circletransform
import trackpy as tp
import numpy as np


def localize(image, tp_opts=None, **kwargs):
    '''
    Localize features in normalized holographic microscopy images

    Parameters
    ----------
    image: array_like
        image data

    Keyword Parameters
    ------------------
    nfringes: int
        number of fringes to count
    maxrange: int
        maximum extent of a feature [pixels]
    tp_opts: dict
        options for trackpy.localize
        Default: {'diameter': 31, 'minmass': 30}

    Returns
    -------
    centers: numpy.array
        (x, y) coordinates of feature centers
    bboxes: tuple
        ((x0, y0), w, h) bounding box of feature

    Notes
    -----
    Uses circletransform to coalesce ring-like features into
    blobs that can be identified and localized by trackpy.
    Counts fringes to determine feature extent
    '''

    a = circletransform(image)
    a /= np.max(a)
    tp_opts = tp_opts or dict(diameter=31, minmass=30)
    features = tp.locate(a, **tp_opts)

    nfeatures = len(features)
    if nfeatures == 0:
        return features

    centers = features[['x', 'y']].to_numpy()
    bboxes = []
    for center in centers:
        extent = feature_extent(image, center, **kwargs)
        r0 = tuple((center - extent/2).astype(int))
        bboxes.append((r0, extent, extent))
    return centers, bboxes


def feature_extent(norm, center,
                   nfringes=None,
                   maxrange=None):
    '''
    Radius of feature based on counting diffraction fringes

    Parameters
    ----------
    norm: array_like
        Normalized image data
    center: tuple
        (x_p, y_p) coordinates of feature center

    Keywords
    --------
    nfringes: int
        Number of fringes to count. Default: 20
    maxrange: int
        Maximum feature extent. Default: 400

    Returns
    -------
    extent: int
        Extent of feature [pixels]
    '''
    nfringes = nfringes or 20
    maxrange = maxrange or 400
    b = aziavg(norm, center) - 1.
    ndx = np.where(np.diff(np.sign(b)))[0] + 1
    extent = maxrange if len(ndx) <= nfringes else ndx[nfringes]
    return extent


def aziavg(data, center):
    '''Azimuthal average of data about center

    Parameters
    ----------
    data: array_like
        image data
    center: tuple
        (x_p, y_p) center of azimuthal average

    Returns
    -------
    avg: array_like
        One-dimensional azimuthal average of data about center
    '''
    x_p, y_p = center
    ny, nx = data.shape
    x = np.arange(nx) - x_p
    y = np.arange(ny) - y_p

    d = data.ravel()
    r = np.hypot.outer(y, x).astype(np.int).ravel()
    nr = np.bincount(r)
    avg = np.bincount(r, d) / nr
    return avg
