from pylorenzmie.theory import Instrument, LorenzMie
from pylorenzmie.analysis import Localizer, Feature
from pylorenzmie.analysis.Hologram import Hologram
from pylorenzmie.lib import LMObject
from pylorenzmie.lib.lmtypes import Image, Properties, Results
import pandas as pd
import numpy as np


class Frame(LMObject):
    '''Full-frame holographic microscopy analysis pipeline.

    Manages detection, estimation, and optimization of all particle
    features in a single normalized hologram.  Instrument settings
    are shared across all features detected in the frame.

    Inherits from :class:`pylorenzmie.lib.LMObject`.

    Parameters
    ----------
    instrument : Instrument, optional
        Microscope parameters shared by all features.
        Default: ``Instrument()``.
    localizer : Localizer, optional
        Feature detection backend.  Default: ``Localizer()``.
    data : numpy.ndarray, optional
        Normalized hologram.  Setting data clears any previous
        detection results.

    Attributes
    ----------
    shape : tuple[int, int]
        Height × width of the most recently assigned image.
    coordinates : numpy.ndarray
        Pixel coordinate grid of shape ``(2, height, width)``.
    features : list[Feature]
        Feature objects created by the most recent :meth:`detect` call.
    bboxes : list
        Bounding boxes ``((x0, y0), w, h)`` for current features.
    results : pandas.DataFrame
        Tracking and characterization results from the most recent
        :meth:`optimize` or :meth:`analyze` call.
    '''

    def __init__(self,
                 instrument: Instrument | None = None,
                 localizer: Localizer | None = None,
                 data: Image | None = None) -> None:
        super().__init__()
        self.instrument = instrument or Instrument()
        self.localizer = localizer or Localizer()
        self._shape: tuple[int, int] = (0, 0)
        self._features: list[Feature] = []
        self._bboxes: list = []
        self._results: pd.DataFrame = pd.DataFrame()
        self._coordinates: np.ndarray = np.empty((2, 0, 0))
        self._data: Image | None = None
        if data is not None:
            self.data = data

    @LMObject.properties.getter
    def properties(self) -> Properties:
        return dict()

    @property
    def shape(self) -> tuple[int, int]:
        '''Height × width of the current image.'''
        return self._shape

    @shape.setter
    def shape(self, shape: tuple[int, int] | None) -> None:
        if shape is None:
            return
        shape = tuple(shape)
        if shape == self._shape:
            return
        self._coordinates = self.meshgrid(shape, flatten=False)
        self._shape = shape

    @property
    def coordinates(self) -> np.ndarray:
        '''Pixel coordinate grid, shape ``(2, height, width)``.'''
        return self._coordinates

    @property
    def data(self) -> Image | None:
        '''Normalized hologram intensity.'''
        return self._data

    @data.setter
    def data(self, data: Image | None) -> None:
        self._features = []
        self._results = pd.DataFrame()
        if data is not None:
            self.shape = data.shape
        self._data = data

    @property
    def results(self) -> Results:
        '''Tracking and characterization results from the most recent fit.'''
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
        for bbox in self._bboxes:
            (x0, y0), w, h = bbox
            dim = min(w, h)
            d = self.data[y0:y0 + dim, x0:x0 + dim]
            holo = Hologram(d, corner=(float(x0), float(y0)))
            this = Feature(hologram=holo,
                           model=LorenzMie(instrument=self.instrument))
            this.particle.x_p = x0 + dim / 2.
            this.particle.y_p = y0 + dim / 2.
            self._features.append(this)

    def detect(self) -> int:
        '''Detect and localize features in :attr:`data`.

        Returns
        -------
        nfeatures : int
            Number of features found.
        '''
        if self.data is None:
            self._features = []
            self._bboxes = []
            return 0
        self._results = self.localizer.localize(self.data)
        self._features = []
        self._bboxes = []
        for _, row in self._results.iterrows():
            (x0, y0), w, h = row.bbox
            dim = min(w, h)
            d = self.data[y0:y0 + dim, x0:x0 + dim]
            holo = Hologram(d, corner=(float(x0), float(y0)))
            this = Feature(hologram=holo,
                           model=LorenzMie(instrument=self.instrument))
            this.particle.x_p = row.x_p
            this.particle.y_p = row.y_p
            self._features.append(this)
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
        frame.data = example_hologram('image0010.png')

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
