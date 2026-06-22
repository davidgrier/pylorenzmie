from pylorenzmie.lib import LMObject, Azimuthal, CircleTransform
from pylorenzmie.lib.lmtypes import Image, Properties
import numpy as np
import trackpy as tp
import pandas as pd


class Localizer(LMObject):
    '''Detect and localize holographic features in a normalized image.

    Uses :class:`~pylorenzmie.lib.CircleTransform` to enhance ring-like
    patterns, then calls :func:`trackpy.locate` to find feature centers.
    Bounding boxes are sized to enclose a specified number of fringes.

    Inherits from :class:`pylorenzmie.lib.LMObject`.

    Parameters
    ----------
    diameter : int, optional
        Nominal feature diameter passed to ``trackpy.locate`` [pixels].
        Default: 31.
    nfringes : int, optional
        Number of interference fringes to enclose in each bounding box.
        Default: 20.
    '''

    def __init__(self,
                 diameter: int = 31,
                 nfringes: int = 20) -> None:
        super().__init__()
        self.diameter = diameter
        self.nfringes = nfringes
        self._circletransform = CircleTransform()

    @LMObject.properties.getter
    def properties(self) -> Properties:
        '''Localization parameters: nfringes and diameter.'''
        return dict(nfringes=self.nfringes, diameter=self.diameter)

    def localize(self,
                 image: Image | list[Image],
                 diameter: int | None = None,
                 nfringes: int | None = None,
                 **kwargs) -> pd.DataFrame:
        '''Localize features in a normalized holographic microscopy image.

        Parameters
        ----------
        image : numpy.ndarray or list of numpy.ndarray
            Normalized hologram intensity.
        diameter : int, optional
            Override :attr:`diameter` for this call.
        nfringes : int, optional
            Override :attr:`nfringes` for this call.
        **kwargs
            Additional keyword arguments passed to ``trackpy.locate``.

        Returns
        -------
        predictions : pandas.DataFrame
            Columns: ``x_p``, ``y_p``, ``bbox``.
            ``bbox`` is ``((x0, y0), width, height)``.
        '''
        if isinstance(image, list):
            return [self.localize(b, diameter, nfringes, **kwargs)
                    for b in image]

        diameter = self.diameter if diameter is None else diameter
        nfringes = self.nfringes if nfringes is None else nfringes

        a = self._circletransform.transform(image)
        features = tp.locate(a, diameter, characterize=False, **kwargs)

        predictions = []
        for _, feature in features.iterrows():
            r_p = feature[['x', 'y']].to_numpy()
            b = Azimuthal.avg(image, r_p)
            ndx = np.where(np.diff(np.sign(b - 1.)) != 0)[0] + 1
            try:
                extent = ndx[nfringes]
            except IndexError:
                extent = ndx[-1]
            r0 = tuple((r_p - extent / 2).astype(int))
            bbox = (r0, extent, extent)
            predictions.append(dict(x_p=r_p[0], y_p=r_p[1], bbox=bbox))
        return pd.DataFrame(predictions)

    detect = localize

    def crop(self,
             image: Image,
             bbox: tuple[tuple[int, int], int, int]) -> Image:
        '''Crop an image to a bounding box.

        Parameters
        ----------
        image : numpy.ndarray
            Full-frame image.
        bbox : tuple
            ``((x0, y0), width, height)``.

        Returns
        -------
        cropped : numpy.ndarray
            Sub-image of shape ``(height, width)``.
        '''
        (x0, y0), w, h = bbox
        return image[y0:y0 + h, x0:x0 + w]

    @classmethod
    def example(cls) -> None:  # pragma: no cover
        from time import perf_counter
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from pylorenzmie.utilities import example_hologram

        localizer = cls()
        image = example_hologram('image0010.png').data

        start = perf_counter()
        features = localizer.localize(image)
        print(f'Elapsed time: {perf_counter() - start:.3f} s')
        print(features)

        style = dict(fill=False, linewidth=3, edgecolor='r')
        fig, ax = plt.subplots()
        ax.imshow(image, cmap='gray')
        for bbox in features.bbox:
            ax.add_patch(Rectangle(*bbox, **style))
        plt.show()


if __name__ == '__main__':  # pragma: no cover
    Localizer.example()
