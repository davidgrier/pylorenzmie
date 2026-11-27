'''Abstract base class for hologram parameter estimators.'''

from abc import abstractmethod
from typing import ClassVar

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from pylorenzmie.lib import LMObject
from pylorenzmie.analysis.Hologram import Hologram


_NPIXELS = 20000      # pixels used to find the center
_RING = 0.5           # width (pixels) of the rings used to find the center
_NEAR = 5.            # search (pixels) about where a particle was seen


class BaseEstimator(LMObject):
    '''Abstract base for particle parameter estimators.

    Subclasses must implement :meth:`estimate` and the
    :attr:`~pylorenzmie.lib.LMObject.LMObject.properties` getter inherited
    from :class:`~pylorenzmie.lib.LMObject`.

    See Also
    --------
    ~pylorenzmie.analysis.Estimator.Estimator : conventional azimuthal-profile
        estimator.
    ~pylorenzmie.analysis.DEEstimator.DEEstimator : global
        differential-evolution estimator.

    Notes
    -----
    The estimate includes the center of the particle, ``x_p`` and ``y_p``,
    for which :meth:`find_center` serves.

    :class:`~pylorenzmie.analysis.Feature.Feature` sets :attr:`exclude` to
    the pixels that its mask leaves out, and :attr:`center` to where the
    particle was seen, if it was told.  An estimator that has no use for
    them ignores them.
    '''

    exclude: NDArray[np.bool_] | None = None
    '''Pixels that the estimate must not use, or ``None``.'''

    center: tuple[float, float] | None = None
    '''Where the particle was seen, ``(x_p, y_p)`` in pixels, as by a
    localizer, or ``None``.  ``find_center`` starts there.'''

    max_shift: float | None = None
    '''Largest distance (pixels) along each axis by which ``find_center``
    moves the center from its start, never more than a quarter of the smaller
    side of the crop.  ``None`` is 5 from ``center``, and from the middle of
    the crop ``unhinted_shift``; 0 keeps the start.'''

    unhinted_shift: ClassVar[float] = 0.
    '''The search (pixels) about the middle of the crop when there is no
    ``center``.  0, because a wide search can leave the middle of a crop
    that holds two particles for one of them.'''

    def find_center(self, hologram: Hologram) -> tuple[float, float]:
        '''Center of the most rotationally symmetric pattern near a start.

        The start is :attr:`center`, or without it the middle of the crop,
        which by default is then returned as it is.  The center that is
        returned minimizes the variance of the pixels
        within rings of 0.5 pixel about it, searched on grids of decreasing
        size and spacing within :attr:`max_shift` of the start.  It needs
        neither the radius nor the height of the particle.

        Parameters
        ----------
        hologram : Hologram
            Normalized hologram crop.

        Returns
        -------
        center : tuple[float, float]
            ``(x_p, y_p)``, in the pixel coordinates of the hologram.

        Notes
        -----
        On crops of frames of silica spheres, the search placed the center
        within 3 pixels of that of a fit for 91% of the particles that the
        built-in localizer found, starting from its positions, against 57%
        for the positions themselves.  The patterns of neighbors can mislead
        it, and a wide search without a good start can find a neighbor.  From
        positions that were already good, those of the CATCH detector, it
        improved the typical error and made large errors a little more
        frequent.
        '''
        if self.center is None:
            x = float(hologram.coordinates[0].mean())
            y = float(hologram.coordinates[1].mean())
            reach = self.unhinted_shift
        else:
            x, y = (float(value) for value in self.center)
            reach = _NEAR
        if self.max_shift is not None:
            reach = self.max_shift
        # a small crop says little about a center that is far from its start
        reach = min(reach, min(hologram.data.shape) / 4.)
        if reach <= 0.:
            return x, y
        data = hologram.flat_data
        xy = hologram.flat_coordinates
        keep = np.isfinite(data)
        if self.exclude is not None:
            keep &= ~np.asarray(self.exclude, bool).ravel()
        if not keep.any():
            return x, y
        step = max(1, int(np.ceil(keep.sum() / _NPIXELS)))
        pixels = np.flatnonzero(keep)[::step]
        d, px, py = data[pixels], xy[0, pixels], xy[1, pixels]

        def unexplained(cx: float, cy: float) -> float:
            bins = np.floor(np.hypot(px - cx, py - cy) / _RING).astype(int)
            n = np.bincount(bins)
            occupied = n > 0
            total = np.bincount(bins, d)[occupied]
            squares = np.bincount(bins, d * d)[occupied]
            # rounded, so that rounding errors do not decide among equals
            return round(float(np.sum(squares - total**2 / n[occupied])), 8)

        start = x, y
        coarse = 2. if reach > 3. else 1.
        for span, spacing in ((reach, coarse), (min(2., reach), 1.),
                              (min(1., reach), 0.25)):
            steps_out = int(span / spacing)
            offsets = spacing * np.arange(-steps_out, steps_out + 1)
            # nearest first: data that cannot tell, such as a uniform
            # image, leave the center where it was
            steps = sorted(((dx, dy) for dx in offsets for dy in offsets),
                           key=lambda s: s[0]**2 + s[1]**2)
            trials = [(x + dx, y + dy) for dx, dy in steps
                      if abs(x + dx - start[0]) <= reach
                      and abs(y + dy - start[1]) <= reach]
            x, y = min(trials, key=lambda c: unexplained(*c))
        return float(x), float(y)

    @abstractmethod
    def estimate(self, hologram: Hologram) -> pd.Series:
        '''Estimate particle parameters from a normalized hologram.

        Parameters
        ----------
        hologram : Hologram
            Normalized hologram crop to analyze.

        Returns
        -------
        result : pandas.Series
            Estimated particle properties.

        Notes
        -----
        The result is the estimate.  A
        :class:`~pylorenzmie.analysis.Feature.Feature` writes it onto the
        particle of its model, so an estimator need not set the particle
        itself (and one that has no model cannot).
        '''
