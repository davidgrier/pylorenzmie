from pylorenzmie.lib import (LMObject, aziavg, CircleTransform)
import numpy as np
import trackpy as tp
import pandas as pd
from typing import Optional


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
                 diameter: Optional[int] = None,
                 nfringes: Optional[int] = None,
                 maxrange: Optional[int] = None,
                 **kwargs) -> None:
        self.diameter = diameter or 31
        self.nfringes = nfringes or 20
        self.maxrange = maxrange or 400
        self._circletransform = CircleTransform()
        self.detect = self.localize

    @LMObject.properties.fget
    def properties(self) -> dict:
        keys = 'nfringes maxrange diameter'.split()
        return {k: getattr(self, k) for k in keys}

    def localize(self,
                 image: np.ndarray,
                 diameter: Optional[int] = None,
                 nfringes: Optional[int] = None,
                 **kwargs) -> pd.DataFrame:
        '''
        Localize features in normalized holographic microscopy images

        Parameters
        ----------
        image : numpy.ndarray
            image data
        diameter : Optional[int]
            typical size of feature [pixels]
        nfringes : Optional[int]
            number of fringes to enclose in bounding box

        localize() also accepts all of the keyword arguments
        for trackpy.locate()

        Returns
        -------
        results: pandas.DataFrame
           x_p, y_p, bbox
           bbox: ((x0, y0), w, h)
        '''
        diameter = diameter or self.diameter
        nfringes = nfringes or self.nfringes

        a = self._circletransform.transform(image)
        features = tp.locate(a, diameter, characterize=False, **kwargs)

        predictions = []
        for n, feature in features.iterrows():
            r_p = feature[['x', 'y']]
            b = aziavg(image, r_p) - 1.
            ndx = np.where(np.diff(np.sign(b)))[0] + 1
            toobig = len(ndx) <= nfringes
            extent = self.maxrange if toobig else ndx[nfringes]
            r0 = tuple((r_p - extent/2).astype(int))
            bbox = (r0, extent, extent)
            prediction = dict(x_p=r_p[0], y_p=r_p[1], bbox=bbox)
            predictions.append(prediction)
        return pd.DataFrame(predictions)


def example():
    import cv2
    import matplotlib
    from matplotlib import pyplot as plt
    from matplotlib.patches import Rectangle
    from pathlib import Path

    # Create a Localizer
    localizer = Localizer()

    # Normalized hologram
    basedir = Path(__file__).parent.parent.resolve()
    filename = str(basedir / 'docs' / 'tutorials'/ 'PS_silica.png')
    b = cv2.imread(filename, cv2.IMREAD_GRAYSCALE).astype(float) / 100.
    print(filename)

    # Use Localizer to identify features in the hologram
    features = localizer.localize(b)
    print(features)

    # Show and report results
    style = dict(fill=False, linewidth=3, edgecolor='r')
    matplotlib.use('TkAgg')
    fig, ax = plt.subplots()
    ax.imshow(b, cmap='gray')
    for bbox in features.bbox:
        ax.add_patch(Rectangle(*bbox, **style))
    plt.show()


if __name__ == '__main__':
    example()
