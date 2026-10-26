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
'''

import numpy as np
import pandas as pd
from pylorenzmie.lib import meshgrid
from pylorenzmie.theory import Instrument, LorenzMie
from pylorenzmie.analysis import Localizer, Feature
from pylorenzmie.analysis.Hologram import Hologram
from pylorenzmie.analysis.pair_grouping import group_overlapping
from pylorenzmie.lib.lmtypes import Image, Results


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
        Microscope parameters shared by all features.
        Default: ``Instrument()``.
    localizer : Localizer, optional
        Feature detection backend.  Default: ``Localizer()``.
    split_redchi : float or None, optional
        Reduced chi-squared above which a single-particle fit is
        suspected to be two particles and is refit as a
        :class:`~pylorenzmie.theory.Pair`.  The pair fit replaces the
        single fit only if it has a lower redchi.  ``None`` disables
        this check.  Default: 2.0.

    Attributes
    ----------
    features : list[Feature]
        Feature objects created by the most recent :meth:`detect` call.
    bboxes : list
        Bounding boxes ``((x0, y0), w, h)`` for current features.
    results : pandas.DataFrame
        Optimized parameters from the most recent :meth:`optimize` or
        :meth:`analyze` call.
    '''

    def __init__(self,
                 data: Image | None = None,
                 corner: tuple[float, float] = (0., 0.),
                 instrument: Instrument | None = None,
                 localizer: Localizer | None = None,
                 split_redchi: float | None = 2.0) -> None:
        self.instrument = instrument or Instrument()
        self.localizer = localizer or Localizer()
        self.split_redchi = split_redchi
        self._features: list[Feature] = []
        self._bboxes: list = []
        self._results: pd.DataFrame = pd.DataFrame()
        # Initialize Hologram-level state directly; super().__init__() is
        # skipped so that data=None is valid before any image is loaded.
        self.corner = corner
        self._coordinates = None
        self._data: Image | None = None
        if data is not None:
            self.data = data

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
        self._results = pd.DataFrame()
        self._data = data
        if data is not None:
            self._coordinates = meshgrid(data.shape,
                                         corner=self.corner,
                                         flatten=False)
        else:
            self._coordinates = None

    @property
    def shape(self) -> tuple[int, int]:
        '''Image dimensions ``(ny, nx)``, or ``(0, 0)`` if no data.'''
        if self._data is None:
            return (0, 0)
        return self._data.shape

    @property
    def results(self) -> Results:
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
        self._features = []
        for (x0, y0), w, h in self._bboxes:
            dim = min(w, h)
            feature = self[y0:y0 + dim, x0:x0 + dim]
            feature.particle.x_p = x0 + dim / 2.
            feature.particle.y_p = y0 + dim / 2.
            self._features.append(feature)

    def _feature(self,
                 key: tuple,
                 centers: list[tuple[float, float]] | None = None) -> Feature:
        '''Build a Feature crop with matching corner and shared instrument.'''
        return Feature(Hologram.__getitem__(self, key),
                       model=LorenzMie(instrument=self.instrument),
                       centers=centers)

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

    def detect(self) -> int:
        '''Detect and localize features in :attr:`data`.

        Localized bounding boxes are passed through
        :func:`~pylorenzmie.analysis.pair_grouping.group_overlapping`:
        an isolated box becomes a single-particle :class:`Feature`,
        while two mutually overlapping boxes are merged into one
        :class:`Feature` configured for a
        :class:`~pylorenzmie.theory.Pair`.

        Returns
        -------
        nfeatures : int
            Number of features found.
        '''
        if self._data is None:
            self._features = []
            self._bboxes = []
            return 0
        df = group_overlapping(self.localizer.localize(self._data))
        self._features = []
        self._bboxes = []
        for _, row in df.iterrows():
            (x0, y0), w, h = row.bbox
            x0, y0 = max(int(x0), 0), max(int(y0), 0)
            if row.n_particles == 2:
                feature = self._feature(
                    (slice(y0, y0 + h), slice(x0, x0 + w)),
                    centers=list(zip(row.x_p, row.y_p)))
            else:
                dim = min(w, h)
                feature = self._feature(
                    (slice(y0, y0 + dim), slice(x0, x0 + dim)))
                feature.particle.x_p = row.x_p
                feature.particle.y_p = row.y_p
            self._features.append(feature)
            self._bboxes.append(row.bbox)
        return len(self._features)

    def estimate(self) -> None:
        '''Estimate parameters for all current features.'''
        for feature in self.features:
            feature.estimate()

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
        that separated is normally resolved as two boxes by
        :func:`~pylorenzmie.analysis.pair_grouping.group_overlapping`
        before ever reaching this single-to-pair promotion, so a
        feature that got this far is already blended tightly enough
        that 1/4 of the crop size overshoots past any separation it
        is likely to have.

        Parameters
        ----------
        feature : Feature
            Single-particle feature that has already been optimized.

        Returns
        -------
        pair : Feature
            Optimized pair feature with the lowest redchi.
        '''
        fit = feature.optimizer.result
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
                pair.estimator.seed = feature.estimator.seed
                pair.estimate()
                pair.optimize()
                if (best is None or
                        pair.optimizer.result.redchi <
                        best.optimizer.result.redchi):
                    best = pair
        return best

    def split_suspects(self) -> int:
        '''Refit suspect single-particle features as two-particle Pairs.

        A single-particle feature whose redchi exceeds
        :attr:`split_redchi` is refit as a
        :class:`~pylorenzmie.theory.Pair` (see :meth:`_split`).  The
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
            result = feature.optimizer.result
            if feature.centers is None and result.redchi > self.split_redchi:
                pair = self._split(feature)
                if pair.optimizer.result.redchi < result.redchi:
                    self._features[i] = pair
                    nsplit += 1
        return nsplit

    def optimize(self) -> Results:
        '''Optimize parameters for all current features.

        Suspect single-particle fits are then refit as pairs, as in
        :meth:`split_suspects`.

        Returns
        -------
        results : pandas.DataFrame
            Fitted values, uncertainties, and goodness-of-fit for each
            feature.
        '''
        for feature in self.features:
            feature.optimize()
        self.split_suspects()
        self._results = pd.DataFrame(
            [feature.optimizer.result for feature in self.features])
        return self._results

    def analyze(self, data: Image | None = None) -> Results:
        '''Detect features, estimate parameters, and optimize fits.

        Parameters
        ----------
        data : numpy.ndarray, optional
            Normalized hologram.  Updates :attr:`data` if provided.

        Returns
        -------
        results : pandas.DataFrame
            Optimized parameters of the generative model for each
            detected feature.
        '''
        self.data = data
        self.detect()
        self.estimate()
        return self.optimize()

    @classmethod
    def example(cls) -> None:  # pragma: no cover
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
