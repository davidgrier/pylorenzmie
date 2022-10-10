from pylorenzmie.lib import LMObject
import numpy as np
import trackpy as tp
from pylorenzmie.utilities import (aziavg, Circletransform)


class Localizer(LMObject):
    '''Identify and localize features in holograms

    Properties
    ----------
    nfringes : int
        Number of interference fringes used to determine feature extent
        Default: 20
    maxrange : int
        Maximum extent of feature [pixels]
        Default: 400
    diameter : int
        Scale of localized features [pixels]
        Default: 31

    Methods
    -------
    detect(image) : list of dict
        Returns centers and bounding boxes of features
        detected in image
    '''

    def __init__(self,
                 diameter=None,
                 nfringes=None,
                 maxrange=None,
                 **kwargs):
        self.diameter = diameter or 31
        self.nfringes = nfringes or 20
        self.maxrange = maxrange or 400
        self._circletransform = Circletransform()

    @LMObject.properties.fget
    def properties(self) -> dict:
        keys = 'nfringes maxrange diameter'.split()
        return {k: getattr(self, k) for k in keys}

    def detect(self, image, **kwargs):
        '''
        Localize features in normalized holographic microscopy images

        Parameters
        ----------
        image : array_like
            image data

        detect() also accepts all of the keyword arguments
        for trackpy.locate()

        Returns
        -------
        centers : numpy.array
            (x, y) coordinates of feature centers
        bboxes : tuple
            ((x0, y0), w, h) bounding box of feature
        '''
        a = self._circletransform.transform(image)
        features = tp.locate(a, self.diameter, characterize=False, **kwargs)

        predictions = []
        for n, feature in features.iterrows():
            r_p = feature[['x', 'y']]
            extent = self._extent(image, r_p)
            r0 = tuple((r_p - extent/2).astype(int))
            bbox = (r0, extent, extent)
            prediction = dict(x_p=r_p[0], y_p=r_p[1], bbox=bbox)
            predictions.append(prediction)
        return predictions

    def _extent(self, data, center):
        '''Return radius of feature by counting diffraction fringes

        Parameters
        ----------
        data : array_like
            Normalized image data
        center : tuple
            (x_p, y_p) coordinates of feature center

        Returns
        -------
        extent : int
            Extent of feature [pixels]
        '''
        b = aziavg(data, center) - 1.
        ndx = np.where(np.diff(np.sign(b)))[0] + 1
        toobig = len(ndx) <= self.nfringes
        return self.maxrange if toobig else ndx[self.nfringes]
