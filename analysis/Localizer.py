from pylorenzmie.lib import (LMObject, aziavg, CircleTransform)
import numpy as np
from numpy.typing import NDArray
import trackpy as tp
import pandas as pd


Image = NDArray[int | float]
Prediction = pd.DataFrame


class Localizer(LMObject):
    '''Identify and localize features in holograms

    Properties
    ----------
    nfringes : int
        Number of interference fringes used to determine feature extent
        Default: 20
    diameter : int
        Scale of localized features [pixels]
        Default: 31

    Methods
    -------
    localize(image) : pandas.DataFrame | list[pandas.DataFrame]
        Returns centers and bounding boxes of features
        detected in image

        Arguments
        ---------
        image: numpy.ndarray | list[numpy.ndarray]
        diameter : int
        nfringes : int

    detect :
        Synonym for localize for backward compatibility
    '''

    def __init__(self,
                 diameter: int | None = None,
                 nfringes: int | None = None,
                 **kwargs) -> None:
        self.diameter = diameter or 31
        self.nfringes = nfringes or 20
        self._circletransform = CircleTransform()
        self.detect = self.localize

    @LMObject.properties.fget
    def properties(self) -> dict:
        keys = 'nfringes diameter'.split()
        return {k: getattr(self, k) for k in keys}

    def localize(self,
                 image: Image | list[Image],
                 diameter: int | None = None,
                 nfringes: int | None = None,
                 **kwargs) -> Prediction:
        '''
        Localize features in normalized holographic microscopy images

        Arguments
        ---------
        image : numpy.ndarray | list[numpy.ndarray]
            image data
        diameter : int
            typical size of feature [pixels]
        nfringes : int
            number of fringes to enclose in bounding box

        localize() also accepts all of the keyword arguments
        for trackpy.locate()

        Returns
        -------
        predictions: pandas.DataFrame | list[pandas.DataFrame]
           x_p, y_p, bbox
           bbox: ((x0, y0), w, h)
        '''
        diameter = diameter or self.diameter
        nfringes = nfringes or self.nfringes

        if isinstance(image, list):
            return [self.localize(b, diameter, nfringes) for b in image]

        a = self._circletransform.transform(image)
        features = tp.locate(a, diameter, characterize=False, **kwargs)

        predictions = []
        for n, feature in features.iterrows():
            r_p = feature[['x', 'y']].to_numpy()
            b = aziavg(image, r_p)
            p = b > 1.
            ndx = np.where(p[1:] ^ p[:-1])[0] + 1
            extent = ndx[nfringes] if len(ndx) > nfringes else ndx[-1]
            r0 = tuple((r_p - extent/2).astype(int))
            bbox = (r0, extent, extent)
            prediction = dict(x_p=r_p[0], y_p=r_p[1], bbox=bbox)
            predictions.append(prediction)
        return pd.DataFrame(predictions)


def example() -> None:
    import cv2
    import matplotlib
    from matplotlib import pyplot as plt
    from matplotlib.patches import Rectangle

    # Create a Localizer
    localizer = Localizer()

    # Normalized hologram
    basedir = localizer.directory.parent
    filename = str(basedir / 'docs' / 'tutorials' / 'image0010.png')
    b = cv2.imread(filename, cv2.IMREAD_GRAYSCALE).astype(float) / 100.
    print(filename)

    # Use Localizer to identify features in the hologram
    features = localizer.localize(b, nfringes=20)
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
