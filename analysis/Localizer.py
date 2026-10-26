'''Detection of holographic features in a frame.

A :class:`Localizer` finds the particles in a normalized image.  It enhances
the ring-like patterns of the holograms with a circle transform
(:class:`~pylorenzmie.lib.CircleTransform.CircleTransform`), locates the
centers of the bright spots with :func:`trackpy.locate`, and sizes a bounding
box around each center to enclose a chosen number of interference fringes,
clipped to the image.  The result is a table with the position and the box of
each feature.

Detection depends on the nominal ``diameter`` of a feature, in pixels, and the
default can find one particle twice or miss a faint one; the single-frame
tutorial shows how to tune it.  Further keyword arguments of
:meth:`~Localizer.localize` go to :func:`trackpy.locate`.
:meth:`~Localizer.crop` cuts the box out of the image.
'''

from collections.abc import Callable

from pylorenzmie.lib import LMObject, Azimuthal, CircleTransform
from pylorenzmie.lib.lmtypes import Image, Properties
import numpy as np
import trackpy as tp
import pandas as pd


class Localizer(LMObject):
    '''Detect and localize holographic features in a normalized image.

    Uses :class:`~pylorenzmie.lib.CircleTransform` to enhance ring-like
    patterns, then calls :func:`trackpy.locate` to find feature centers.
    Bounding boxes are centered on the feature and sized to enclose a
    specified number of fringes, and are clipped to the image.

    A fringe is one bright and one dark ring, that is, two crossings of
    the azimuthally averaged intensity through 1.  The box extends from
    the center to the radius of the ``2 * nfringes``-th crossing (or to
    the outermost crossing, if there are fewer).

    Inherits from :class:`pylorenzmie.lib.LMObject`.

    Parameters
    ----------
    diameter : int, optional
        Nominal feature diameter passed to ``trackpy.locate`` [pixels].
        Default: 31.
    nfringes : int, optional
        Number of interference fringes to enclose in each bounding box.
        Default: 4.
    percentile : int, optional
        ``trackpy.locate`` brightness cutoff: a detected peak must be
        brighter than pixels in this percentile.  Lower values admit
        fainter (e.g. more out-of-focus) features.  Default: 64
        (``trackpy.locate``'s own default).
    separation : float, optional
        Minimum pixel separation ``trackpy.locate`` requires between
        two detections.  Default: ``None`` (``trackpy.locate``'s own
        default, ``diameter`` + 1).
    '''

    def __init__(self,
                 diameter: int = 31,
                 nfringes: int = 4,
                 percentile: int = 64,
                 separation: float | None = None) -> None:
        super().__init__()
        self.diameter = diameter
        self.nfringes = nfringes
        self.percentile = percentile
        self.separation = separation
        self._circletransform = CircleTransform()

    @LMObject.properties.getter
    def properties(self) -> Properties:
        '''Localization parameters: nfringes, diameter, percentile,
        and separation.'''
        return dict(nfringes=self.nfringes, diameter=self.diameter,
                    percentile=self.percentile, separation=self.separation)

    def localize(self,
                 image: Image | list[Image],
                 diameter: int | None = None,
                 nfringes: int | None = None,
                 maxfeatures: int | None = None,
                 check: Callable[[], None] | None = None,
                 percentile: int | None = None,
                 separation: float | None = None,
                 **kwargs) -> pd.DataFrame:
        '''Localize features in a normalized holographic microscopy image.

        Parameters
        ----------
        image : numpy.ndarray or list of numpy.ndarray
            Normalized hologram intensity.
        diameter : int, optional
            Override ``diameter`` for this call.
        nfringes : int, optional
            Override ``nfringes`` for this call.
        maxfeatures : int, optional
            Keep only this many features, the strongest ones, strongest
            first.  A noisy image can have hundreds of candidates, and the
            bounding box of each one costs time.  Default: all of them,
            in the order of ``trackpy``.
        check : callable, optional
            Called with no arguments before the bounding box of each feature
            is computed, which is where most of the time goes.  It may raise
            an exception to stop the localization, as a cancel button does.
        percentile : int, optional
            Override :attr:`percentile` for this call.
        separation : float, optional
            Override :attr:`separation` for this call.
        **kwargs
            Additional keyword arguments passed to ``trackpy.locate``.

        Returns
        -------
        predictions : pandas.DataFrame
            Columns: ``x_p``, ``y_p``, ``bbox``, ``mass``, with one row
            per feature; empty if nothing was found.  ``bbox`` is
            ``((x0, y0), width, height)`` in whole pixels, inside the
            image, so a box at the edge of the image is not square.
            ``mass`` is the strength of the feature in the transformed
            image, which ranks candidates.  ``predictions.attrs['nfound']``
            is the number of features that ``trackpy`` found, which is
            larger than ``len(predictions)`` if ``maxfeatures`` applies.
        '''
        if isinstance(image, list):
            return [self.localize(b, diameter, nfringes, maxfeatures, check,
                                  percentile, separation, **kwargs)
                    for b in image]

        diameter = self.diameter if diameter is None else diameter
        nfringes = self.nfringes if nfringes is None else nfringes
        percentile = self.percentile if percentile is None else percentile
        separation = self.separation if separation is None else separation

        a = self._circletransform.transform(image)
        features = tp.locate(a, diameter, characterize=False,
                             percentile=percentile, separation=separation,
                             **kwargs)
        nfound = len(features)
        if maxfeatures is not None and nfound > maxfeatures:
            features = features.nlargest(maxfeatures, 'mass')

        ny, nx = image.shape
        predictions = []
        for _, feature in features.iterrows():
            if check is not None:
                check()
            r_p = feature[['x', 'y']].to_numpy()
            radius = self._extent(Azimuthal.avg(image, r_p), nfringes)
            x0 = max(int(np.floor(r_p[0] - radius)), 0)
            y0 = max(int(np.floor(r_p[1] - radius)), 0)
            x1 = min(int(np.ceil(r_p[0] + radius)) + 1, nx)
            y1 = min(int(np.ceil(r_p[1] + radius)) + 1, ny)
            bbox = ((x0, y0), x1 - x0, y1 - y0)
            predictions.append(dict(x_p=r_p[0], y_p=r_p[1], bbox=bbox,
                                    mass=feature['mass']))
        result = pd.DataFrame(predictions,
                              columns=['x_p', 'y_p', 'bbox', 'mass'])
        result.attrs['nfound'] = nfound
        return result

    @staticmethod
    def _extent(profile: np.ndarray, nfringes: int) -> float:
        '''Radius [pixels] that encloses ``nfringes`` fringes.

        Two crossings of the radial profile through 1 make one fringe.
        If the profile has fewer crossings, the outermost is used, and
        if it has none, the whole profile.
        '''
        crossings = np.where(np.diff(np.sign(profile - 1.)) != 0)[0] + 1
        if crossings.size == 0:
            return float(profile.size - 1)
        return float(crossings[min(2 * nfringes, crossings.size) - 1])

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
