import pandas as pd
from pylorenzmie.lib import meshgrid
from pylorenzmie.theory import Instrument, LorenzMie
from pylorenzmie.analysis import Localizer, Feature
from pylorenzmie.analysis.Hologram import Hologram
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
                 localizer: Localizer | None = None) -> None:
        self.instrument = instrument or Instrument()
        self.localizer = localizer or Localizer()
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
        return Feature(Hologram.__getitem__(self, key),
                       model=LorenzMie(instrument=self.instrument))

    def detect(self) -> int:
        '''Detect and localize features in :attr:`data`.

        Returns
        -------
        nfeatures : int
            Number of features found.
        '''
        if self._data is None:
            self._features = []
            self._bboxes = []
            return 0
        df = self.localizer.localize(self._data)
        self._features = []
        self._bboxes = []
        for _, row in df.iterrows():
            (x0, y0), w, h = row.bbox
            dim = min(w, h)
            feature = self[y0:y0 + dim, x0:x0 + dim]
            feature.particle.x_p = row.x_p
            feature.particle.y_p = row.y_p
            self._features.append(feature)
            self._bboxes.append(row.bbox)
        return len(self._features)

    def estimate(self) -> None:
        '''Estimate parameters for all current features.'''
        for feature in self.features:
            feature.estimate()

    def optimize(self) -> Results:
        '''Optimize parameters for all current features.

        Returns
        -------
        results : pandas.DataFrame
            Fitted values, uncertainties, and goodness-of-fit for each
            feature.
        '''
        results = [feature.optimize() for feature in self.features]
        self._results = pd.DataFrame(results)
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
