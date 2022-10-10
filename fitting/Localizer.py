import numpy as np
import trackpy as tp
from pylorenzmie.utilities import (aziavg, Circletransform)


class Localizer(object):
    '''Identify and localize features in holograms

    Properties
    ----------
    nfringes : int
        Number of interference fringes used to determine feature extent
        Default: 20
    maxrange : int
        Maximum extent of feature [pixels]
        Default: 400
    tp_opts : dict
        Dictionary of options for trackpy.locate()
        Default: dict(diameter=31, minmass=30)

    Methods
    -------
    detect(image) : list of dict
        Returns centers and bounding boxes of features
        detected in image
    '''
    def __init__(self,
                 tp_opts=None,
                 nfringes=None,
                 maxrange=None,
                 **kwargs):
        self._circletransform = Circletransform()
        self._tp_opts = tp_opts or dict(diameter=31, minmass=30)
        self._nfringes = nfringes or 20
        self._maxrange = maxrange or 400
        self._shape = None

    def detect(self, image):
        '''
        Localize features in normalized holographic microscopy images

        Parameters
        ----------
        image : array_like
            image data

        Returns
        -------
        centers : numpy.array
            (x, y) coordinates of feature centers
        bboxes : tuple
            ((x0, y0), w, h) bounding box of feature
        '''
        a = self._circletransform.transform(image)
        features = tp.locate(a, **self._tp_opts, characterize=False)

        predictions = []
        for n, feature in features.iterrows():
            r_p = feature[['x', 'y']]
            extent = self._extent(image, r_p)
            r0 = tuple((r_p - extent/2).astype(int))
            bbox = (r0, extent, extent)
            prediction = dict(x_p=r_p[0], y_p=r_p[1], bbox=bbox)
            predictions.append(prediction)
        return predictions

    def _extent(self, norm, center):
        '''Return radius of feature by counting diffraction fringes

        Parameters
        ----------
        norm : array_like
            Normalized image data
        center : tuple
            (x_p, y_p) coordinates of feature center

        Returns
        -------
        extent : int
            Extent of feature [pixels]
        '''
        b = aziavg(norm, center) - 1.
        ndx = np.where(np.diff(np.sign(b)))[0] + 1
        if len(ndx) <= self._nfringes:
            extent = self._maxrange
        else:
            extent = ndx[self._nfringes]
        return extent
