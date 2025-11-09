from pylorenzmie.lib import (LMObject, Azimuthal, CircleTransform)
import numpy as np
import trackpy as tp
import pandas as pd


Prediction = pd.DataFrame


class Localizer(LMObject):
    '''Identify and localize features in holograms

    Inherits
    --------
    pylorenzmie.lib.LMObject

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
                 diameter: int = 31,
                 nfringes: int = 20,
                 **kwargs) -> None:
        self.diameter = diameter
        self.nfringes = nfringes
        self._circletransform = CircleTransform()
        self.detect = self.localize

    @LMObject.properties.fget
    def properties(self) -> LMObject.Properties:
        keys = 'nfringes diameter'.split()
        return {k: getattr(self, k) for k in keys}

    def localize(self,
                 image: LMObject.Image | list[LMObject.Image],
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
            b = Azimuthal.avg(image, r_p)
            ndx = np.where(np.diff(np.sign(b - 1.)) != 0)[0] + 1
            try:
                extent = ndx[nfringes]
            except IndexError:
                extent = ndx[-1]
            r0 = tuple((r_p - extent/2).astype(int))
            bbox = (r0, extent, extent)
            prediction = dict(x_p=r_p[0], y_p=r_p[1], bbox=bbox)
            predictions.append(prediction)
        return pd.DataFrame(predictions)

    def crop(self,
             image: LMObject.Image,
             bbox: tuple[tuple[int, int], int, int]) -> LMObject.Image:
        '''
        Crop image to bounding box

        Arguments
        ---------
        image : numpy.ndarray
            image data
        bbox : tuple
            ((x0, y0), w, h)

        Returns
        -------
        cropped_image : LMObject.Image
        '''
        (x0, y0), w, h = bbox
        return image[y0:y0+h, x0:x0+w]

    @classmethod
    def example(cls: 'Localizer') -> None:
        import cv2
        import matplotlib
        from matplotlib import pyplot as plt
        from matplotlib.patches import Rectangle
        import time

        # Instantiate a Localizer
        localizer = cls()

        # Normalized hologram
        basedir = localizer.directory.parent
        filename = str(basedir / 'docs' / 'tutorials' / 'image0010.png')
        b = cv2.imread(filename, cv2.IMREAD_GRAYSCALE).astype(float) / 100.
        print(filename)

        # Use Localizer to identify features in the hologram
        start = time.perf_counter()
        features = localizer.localize(b, nfringes=20)
        print(f'Elapsed time: {time.perf_counter() - start:.3f} sec')
        print(features)

        # Show and report results
        style = dict(fill=False, linewidth=3, edgecolor='r')
        fig, ax = plt.subplots()
        ax.imshow(b, cmap='gray')
        for bbox in features.bbox:
            ax.add_patch(Rectangle(*bbox, **style))
        plt.show()


if __name__ == '__main__':
    Localizer.example()
