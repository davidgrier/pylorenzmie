'''Analysis of a full camera frame.

A :class:`Frame` is the whole normalized image of the camera, together with a
:class:`~pylorenzmie.analysis.Localizer.Localizer` and the
:class:`~pylorenzmie.theory.Instrument.Instrument` that the particles share.
It takes three steps, which :meth:`~Frame.analyze` runs in turn:

1. :meth:`~Frame.detect` finds the particles and crops a
   :class:`~pylorenzmie.analysis.Feature.Feature` around each,
2. :meth:`~Frame.estimate` finds starting values for each feature, and
3. :meth:`~Frame.optimize` fits each feature and collects the results in a
   table with one row per particle.

The steps can be run one at a time to adjust the estimator or the optimizer of
the features in between, as the single-frame tutorial does.  Each feature is
fitted on its own, so a particle whose pattern overlaps another's is fitted
with the neighbor's fringes as noise.  The table of a series of frames can be
collected in a :class:`~pylorenzmie.analysis.Trajectory.Trajectory`.

Fitting particles as pairs is something to ask for.  With ``pairs=True``,
two detections whose boxes overlap are fitted together as a
:class:`~pylorenzmie.theory.Pair.Pair`, and with ``split_redchi`` a particle
that a single sphere explains poorly is refit as a pair.  A pair has one row
of the table, with columns of its own (``x_p1``, ``x_p2`` and so on).
'''

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from pylorenzmie.theory import Instrument, LorenzMie, Pair
from pylorenzmie.analysis import (BaseEstimator, CATCHAnalyzer, DEEstimator,
                                  Feature, Localizer)
from pylorenzmie.analysis.Hologram import Hologram
from pylorenzmie.analysis.pair_grouping import group_overlapping
from pylorenzmie.lib import LMObject
from pylorenzmie.lib.lmtypes import Image, Properties


ESTIMATES = ('z_p', 'a_p', 'n_p')
'''Columns of a localizer's table that estimate a particle.'''

REPORTED = ('confidence', 'quality')
'''Columns of a localizer's table that :class:`Frame` adds to its results.'''

TRIAL_NFEV = 100
'''Largest number of evaluations of the residuals (``max_nfev``) in each fit
of a pair that :class:`Frame` tries in place of a single sphere.  A trial that
starts far from any good solution can otherwise take a thousand.'''


def _fit(feature: Feature) -> pd.Series:
    '''The fit of a feature that has been optimized.'''
    result = feature.optimizer.result
    if result is None:
        raise RuntimeError('the feature has not been optimized')
    return result


@dataclass
class _Detected(BaseEstimator):
    '''The estimate that came with a detection.'''

    z_p: float
    a_p: float
    n_p: float

    @LMObject.properties.getter
    def properties(self) -> Properties:
        '''The estimate.'''
        return dict(z_p=self.z_p, a_p=self.a_p, n_p=self.n_p)

    @property
    def complete(self) -> bool:
        '''True if every value of the estimate is a finite number.'''
        return bool(np.isfinite([self.z_p, self.a_p, self.n_p]).all())

    def estimate(self, hologram: Hologram) -> pd.Series:
        '''The estimate of the detector, at the detected position.'''
        x_p, y_p = self.center or (np.nan, np.nan)
        return pd.Series(dict(x_p=x_p, y_p=y_p, **self.properties))


class Frame(Hologram):
    '''Full-frame holographic microscopy analysis pipeline.

    A Frame is a Hologram of the full camera field, augmented with a
    :class:`~pylorenzmie.analysis.Localizer` and a shared
    :class:`~pylorenzmie.theory.Instrument`.  Slicing a Frame returns
    a :class:`~pylorenzmie.analysis.Feature` with the correct corner
    coordinates and instrument already attached.

    Parameters
    ----------
    data : numpy.ndarray, optional
        Normalized hologram.  Setting data clears previous results.
        Default: ``None``.
    corner : tuple[float, float], optional
        Top-left corner of this image within the full camera frame.
        Default: ``(0., 0.)``.
    instrument : Instrument, optional
        Microscope parameters shared by all features.  Default: that of a
        :class:`~pylorenzmie.analysis.CATCHAnalyzer.CATCHAnalyzer` that is
        the localizer, and otherwise ``Instrument()``.  The two must agree.
    localizer : Localizer or CATCHAnalyzer, optional
        Finds the particles: any object whose ``localize(image)`` returns a
        table with the columns ``x_p``, ``y_p`` and ``bbox``.  If the table
        also has ``z_p``, ``a_p`` and ``n_p``, as that of a
        :class:`~pylorenzmie.analysis.CATCHAnalyzer.CATCHAnalyzer` does,
        each fit starts from them.  Default: ``Localizer()``.
    split_redchi : float or None, optional
        Reduced chi-squared above which a single-particle fit is
        suspected to be two particles and is refit as a
        :class:`~pylorenzmie.theory.Pair.Pair`.  The pair fit replaces the
        single fit only if it has a lower redchi, which does not ensure that
        it is physical.  ``None`` (default) disables this check; 2.0 is a
        reasonable threshold.
    estimator : callable, optional
        Makes the estimator of each single-particle feature: it is called
        with no arguments, once for each feature, and returns a
        :class:`~pylorenzmie.analysis.BaseEstimator.BaseEstimator`.  A class
        qualifies.  A factory, and not an estimator, because a feature
        sets the pixel mask of its estimator, so that features cannot share
        one.  Pairs keep their
        :class:`~pylorenzmie.analysis.PairEstimator.PairEstimator`.
        An estimator that is given is used instead of any estimates that
        the localizer returns.  Default: ``None``, those estimates, or
        without them the
        :class:`~pylorenzmie.analysis.DEEstimator.DEEstimator` of a
        :class:`~pylorenzmie.analysis.Feature.Feature`.
    saturated : numpy.ndarray of bool, optional
        Saturated pixels of ``data``, which the fits leave out (see
        :class:`~pylorenzmie.analysis.Hologram.Hologram`).  Setting
        :attr:`data` clears it.  Default: ``None``.
    pairs : bool, optional
        ``True`` makes :meth:`detect` fit two detections whose boxes overlap
        as one :class:`~pylorenzmie.theory.Pair.Pair`
        (see :func:`~pylorenzmie.analysis.pair_grouping.group_overlapping`).
        Default: ``False``, every detection is fitted as a single particle.
    robust : bool, optional
        ``True`` fits each feature with a robust loss (see
        :attr:`~pylorenzmie.analysis.Optimizer.Optimizer.robust`), ``False``
        by least squares.  Default: ``None``, robust if the localizer is a
        :class:`~pylorenzmie.analysis.CATCHAnalyzer.CATCHAnalyzer` and
        ``split_redchi`` is not set.  The optimizer of each feature may be
        changed after :meth:`detect`.  A robust fit reports a smaller
        reduced chi-squared than a least-squares fit of the same data, so
        ``split_redchi`` cannot judge it: :meth:`detect` raises a
        ``ValueError`` if both are asked for.

    Attributes
    ----------
    features : list[Feature]
        Feature objects created by the most recent :meth:`detect` call.
    bboxes : list
        Bounding boxes ``((x0, y0), w, h)`` for current features.
    results : pandas.DataFrame
        Optimized parameters from the most recent :meth:`optimize` or
        :meth:`analyze` call, with the ``confidence`` and ``quality`` of
        each detection if the localizer reports them.  A particle whose
        detection came without a complete estimate is not fitted: its row
        has its position, empty values and ``success`` False.
    '''

    def __init__(self,
                 data: Image | None = None,
                 corner: tuple[float, float] = (0., 0.),
                 instrument: Instrument | None = None,
                 localizer: Localizer | CATCHAnalyzer | None = None,
                 split_redchi: float | None = None,
                 estimator: Callable[[], BaseEstimator] | None = None,
                 saturated: NDArray[np.bool_] | None = None,
                 pairs: bool = False,
                 robust: bool | None = None) -> None:
        if instrument is None and isinstance(localizer, CATCHAnalyzer):
            instrument = localizer.instrument
        self.instrument = instrument or Instrument()
        self.robust = robust
        self.estimator = estimator
        self.localizer = localizer or Localizer()
        self.split_redchi = split_redchi
        self.pairs = pairs
        self._features: list[Feature] = []
        self._bboxes: list = []
        self._results: pd.DataFrame = pd.DataFrame()
        self._reports: list[dict] = []
        self._data: Image | None = None
        # A Frame has no image until data is set; its data property says so.
        super().__init__(cast(Image, data), corner, saturated)

    # Override the dataclass-generated 'data' field with a property so
    # that assigning frame.data triggers coordinate regeneration and a
    # reset of any cached features and results.
    @property
    def data(self) -> Image | None:
        '''Normalized hologram intensity.'''
        return self._data

    @data.setter
    def data(self, data: Image | None) -> None:
        self._features = []
        self._reports = []
        self._results = pd.DataFrame()
        self._data = data
        self._coordinates = None
        self.saturated = None       # it belonged to the previous image

    @property
    def shape(self) -> tuple[int, int]:
        '''Image dimensions ``(ny, nx)``, or ``(0, 0)`` if no data.'''
        if self._data is None:
            return (0, 0)
        return self._data.shape

    @property
    def results(self) -> pd.DataFrame:
        '''Optimized parameters from the most recent fit.'''
        return self._results

    @property
    def features(self) -> list[Feature]:
        '''Features detected in the current frame.'''
        return self._features

    @property
    def bboxes(self) -> list:
        '''Bounding boxes of current features, each ``((x0, y0), w, h)``.'''
        return self._bboxes

    @bboxes.setter
    def bboxes(self, bboxes: tuple | list) -> None:
        if (isinstance(bboxes, tuple) and len(bboxes) == 3
                and isinstance(bboxes[0], tuple)):
            bboxes = [bboxes]
        self._bboxes = list(bboxes)
        self._features = [self[self._slices(bbox)] for bbox in self._bboxes]
        self._reports = [{} for _ in self._bboxes]

    @staticmethod
    def _slices(bbox: tuple) -> tuple[slice, slice]:
        '''Rows and columns of a bounding box, trimmed to the image.'''
        (x0, y0), w, h = bbox
        x0, y0 = int(x0), int(y0)
        return (slice(max(y0, 0), max(y0 + int(h), 0)),
                slice(max(x0, 0), max(x0 + int(w), 0)))

    def _feature(self,
                 key: tuple,
                 centers: list[tuple[float, float]] | None = None,
                 detected: BaseEstimator | None = None) -> Feature:
        '''Build a Feature crop with matching corner and shared instrument.

        *detected* holds the estimate of a detection, for a single particle
        and a frame that was not given an estimator.
        '''
        estimator = None
        single = centers is None or len(centers) == 1
        if single:
            estimator = (self.estimator() if self.estimator is not None
                         else detected)
        feature = Feature(Hologram.__getitem__(self, key),
                          model=LorenzMie(instrument=self.instrument),
                          centers=centers,
                          estimator=estimator)
        if self._robust():
            feature.optimizer.robust = True
        return feature

    def __getitem__(self, key: tuple) -> Feature:
        '''Return a Feature crop with matching corner and instrument.

        Parameters
        ----------
        key : tuple[slice, slice]
            ``(slice_y, slice_x)`` row and column slices.

        Returns
        -------
        feature : Feature
            Cropped Feature with correct corner and shared instrument.
        '''
        return self._feature(key)

    def _robust(self) -> bool:
        '''True if the features are to be fitted with a robust loss.'''
        if self.robust is None:
            return (self.split_redchi is None
                    and isinstance(self.localizer, CATCHAnalyzer))
        return self.robust

    def detect(self) -> int:
        '''Detect and localize features in :attr:`data`.

        Each detection of the localizer becomes a single-particle
        :class:`~pylorenzmie.analysis.Feature.Feature`: the box of the
        detection, trimmed to the image, with the detected position as the
        feature's center, from which its estimator starts.  If ``pairs``
        is ``True``, two detections whose boxes overlap only each other
        (see :func:`~pylorenzmie.analysis.pair_grouping.group_overlapping`)
        become one :class:`~pylorenzmie.analysis.Feature.Feature`
        configured for a :class:`~pylorenzmie.theory.Pair.Pair`.

        Returns
        -------
        nfeatures : int
            Number of features found.
        '''
        self._features = []
        self._bboxes = []
        self._reports = []
        if self._data is None:
            return 0
        if self.robust and self.split_redchi is not None:
            raise ValueError('split_redchi cannot judge robust fits: use '
                             'one or the other')
        if isinstance(self.localizer, CATCHAnalyzer):
            self._check_instrument(self.localizer.instrument)
        self.localizer.saturated = self.saturated
        df = self.localizer.localize(self._data)
        estimated = set(ESTIMATES) <= set(df.columns)
        df = (group_overlapping(df) if self.pairs
              else df.assign(n_particles=1))
        for _, row in df.iterrows():
            detected = None
            if row.n_particles == 2:
                centers = list(zip(row.x_p, row.y_p, strict=True))
            else:
                centers = [(row.x_p, row.y_p)]
                if estimated:
                    detected = _Detected(*(float(cast(float, row[p]))
                                           for p in ESTIMATES))
            self._features.append(self._feature(self._slices(row.bbox),
                                                centers,
                                                detected))
            self._bboxes.append(row.bbox)
            self._reports.append({name: row[name] for name in REPORTED
                                  if name in row})
        return len(self._features)

    def _check_instrument(self, other: Instrument) -> None:
        '''Raise a ValueError if the localizer has other optics.'''
        for name in ('wavelength', 'magnification', 'n_m'):
            mine, theirs = getattr(self.instrument, name), getattr(other, name)
            if not np.isclose(mine, theirs):
                raise ValueError(
                    f'the localizer has {name} = {theirs:g} and the frame '
                    f'{mine:g}: give them the same instrument')

    @staticmethod
    def _fittable(feature: Feature) -> bool:
        '''False for a detection that came without a complete estimate.'''
        estimator = feature.estimator
        return not isinstance(estimator, _Detected) or estimator.complete

    def estimate(self) -> None:
        '''Estimate parameters for all current features.'''
        for feature in self.features:
            feature.estimate()

    @staticmethod
    def _unfitted(feature: Feature) -> pd.Series:
        '''The row of a feature that could not be fitted.'''
        row: dict = {}
        for name in feature.optimizer.variables:
            row.update({name: np.nan, 'd' + name: np.nan})
        x_p, y_p = (feature.centers or [(np.nan, np.nan)])[0]
        row.update(x_p=x_p, y_p=y_p, success=False, npix=0, redchi=np.nan)
        return pd.Series(row)

    def _split(self, feature: Feature) -> Feature:
        '''Refit an optimized single-particle feature as a Pair.

        The single fit's center is taken as the midpoint of the pair.
        Two starting centers are placed on either side of it, at four
        angles and two offsets (1/16 and 1/8 of the crop size).  The
        pair search pins its starting centers, so neither the true
        separation nor the best angle is known in advance; each
        candidate is estimated and optimized, and the one with the
        lowest redchi is returned.  1/16 catches badly-blended pairs
        (true separation well under the ~25%-of-crop-size candidate
        that 1/8 alone produces) where the wider candidate starts
        local refinement too far from the true spacing to converge
        correctly.  A wider 1/4 offset was tried and dropped: a pair
        that separated is normally detected as two particles
        before ever reaching this single-to-pair promotion, so a
        feature that got this far is already blended tightly enough
        that 1/4 of the crop size overshoots past any separation it
        is likely to have.

        Each candidate is fitted with at most :data:`TRIAL_NFEV`
        evaluations.  If the one that is returned stopped there, its
        result is not marked as successful.

        Parameters
        ----------
        feature : Feature
            Single-particle feature that has already been optimized.

        Returns
        -------
        pair : Feature
            Optimized pair feature with the lowest redchi.
        '''
        fit = _fit(feature)
        size = max(feature.shape)
        best = None
        for fraction in (1 / 16, 1 / 8):
            offset = fraction * size
            for angle in np.deg2rad((0., 45., 90., 135.)):
                dx, dy = offset * np.cos(angle), offset * np.sin(angle)
                centers = [(fit.x_p - dx, fit.y_p - dy),
                           (fit.x_p + dx, fit.y_p + dy)]
                pair = Feature(feature,
                               model=LorenzMie(instrument=self.instrument),
                               centers=centers)
                pair.mask = feature.mask
                if (isinstance(feature.estimator, DEEstimator)
                        and isinstance(pair.estimator, DEEstimator)):
                    pair.estimator.seed = feature.estimator.seed
                pair.estimate()
                settings = pair.optimizer.settings
                limit = settings['max_nfev']
                settings['max_nfev'] = TRIAL_NFEV
                pair.optimize()
                settings['max_nfev'] = limit     # for later fits
                if best is None or _fit(pair).redchi < _fit(best).redchi:
                    best = pair
        if best is None:
            raise RuntimeError('no pair was tried')
        return best

    def split_suspects(self) -> int:
        '''Refit suspect single-particle features as two-particle Pairs.

        A single-particle feature whose redchi exceeds
        ``split_redchi`` is refit as a
        :class:`~pylorenzmie.theory.Pair.Pair`.  The
        pair replaces the feature only if its redchi is lower.  Call
        after the features have been optimized.

        Returns
        -------
        nsplit : int
            Number of features replaced by a pair.
        '''
        if self.split_redchi is None:
            return 0
        nsplit = 0
        for i, feature in enumerate(self.features):
            if not self._fittable(feature):
                continue
            result = _fit(feature)
            single = not isinstance(feature.particle, Pair)
            if single and result.redchi > self.split_redchi:
                pair = self._split(feature)
                if _fit(pair).redchi < result.redchi:
                    self._features[i] = pair
                    nsplit += 1
        return nsplit

    def optimize(self) -> pd.DataFrame:
        '''Optimize parameters for all current features.

        If ``split_redchi`` is set, suspect single-particle fits are
        then refit as pairs, as in :meth:`split_suspects`.

        Returns
        -------
        results : pandas.DataFrame
            Fitted values, uncertainties, and goodness-of-fit for each
            feature.
        '''
        for feature in self.features:
            if self._fittable(feature):
                feature.optimize()
        self.split_suspects()
        rows = [feature.optimizer.result if self._fittable(feature)
                else self._unfitted(feature) for feature in self.features]
        results = pd.DataFrame(rows)
        for name in REPORTED:
            if any(name in report for report in self._reports):
                results[name] = [r.get(name) for r in self._reports]
        self._results = results
        return self._results

    def analyze(self,
                data: Image | None = None,
                saturated: NDArray[np.bool_] | None = None
                ) -> pd.DataFrame:
        '''Detect features, estimate parameters, and optimize fits.

        Parameters
        ----------
        data : numpy.ndarray, optional
            Normalized hologram.  Updates :attr:`data` if provided.
        saturated : numpy.ndarray of bool, optional
            Saturated pixels of the hologram.  Updates ``saturated`` if
            provided.

        Returns
        -------
        results : pandas.DataFrame
            Optimized parameters of the generative model for each
            detected feature.
        '''
        if data is not None:
            self.data = data  # pyright: ignore[reportIncompatibleVariableOverride]
        if saturated is not None:
            self.saturated = saturated
        self.detect()
        self.estimate()
        return self.optimize()

    @classmethod
    def example(cls) -> None:  # pragma: no cover
        '''Run a demonstration of the class, for use as a script.'''
        from time import perf_counter
        from pylorenzmie.utilities import example_hologram

        frame = cls()
        frame.instrument.wavelength = 0.447
        frame.instrument.magnification = 0.048
        frame.instrument.n_m = 1.34
        frame.data = example_hologram('image0010.png').data

        n = frame.detect()
        print(f'Detected {n} feature(s)')

        frame.estimate()
        for i, feature in enumerate(frame.features):
            print(f'Feature {i}: {feature.particle}')

        start = perf_counter()
        results = frame.optimize()
        print(f'Optimized in {perf_counter() - start:.3f} s')
        print(results)


if __name__ == '__main__':  # pragma: no cover
    Frame.example()
