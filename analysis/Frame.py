from pylorenzmie.theory import Instrument
from pylorenzmie.analysis import (Localizer, Feature)
from pylorenzmie.lib import (LMObject, coordinates as make_coordinates)
import pandas as pd
import numpy as np
from typing import (Optional, Tuple, List, Dict, TypeVar)


AnyFrame = TypeVar('AnyFrame', bound='Frame')


class Frame(LMObject):
    '''
    Abstraction of a holographic microscopy video frame.
    ...

    Properties
    ----------
    instrument : pylorenzmie.theory.Instrument
        All of the features in a holographic image are interpreted
        in the context of the properties of the instrument that
        recorded the frame.
    data : numpy.ndarray
        (w, h) normalized holographic microscopy image
    shape : tuple
        dimensions of image data, updated to reflect most recent image
    coordinates : numpy.ndarray
        [3, w, h] of pixel coordinates for most recent image
    features : list
        List of Feature objects identified in data by detect()
        or analyze()
    results : pandas.DataFrame
        Summary of tracking and characterization data from estimate(),
        optimize() or analyze()

    Methods
    -------
    detect() :
        Detect and localize features in data. Sets features.

        Returns
        -------
        self

    estimate() :
        Estimate particle position and characteristics for each feature.

        Returns
        -------
        self

    optimize() :
        Refine estimates for particle positions and characteristics

        Returns
        -------
        results: pandas.DataFrame
            Summary of tracking and characterization results from data

    analyze([image]) :
        Identify features in image that are associated with
        particles and optimize the parameters of those features.
        Results are obtained by running detect(), estimate() and optimize()

        Arguments
        ---------
        image : [optional] numpy.ndarray
            Image data to analyze. Sets self.data if provided.
            Default: self.data

        Returns
        -------
        results: pandas.DataFrame
            Summary of tracking and characterization results from data
    '''
    def __init__(self,
                 instrument: Optional[Instrument] = None,
                 localizer: Optional[Localizer] = None,
                 data: Optional[np.ndarray] = None) -> None:
        self.instrument = instrument or Instrument()
        self.localizer = localizer or Localizer()
        self._shape = (0, 0)
        self._data = data

    @LMObject.properties.fget
    def properties(self) -> Dict:
        return dict()

    @property
    def shape(self) -> Tuple:
        '''image shape'''
        return self._shape

    @shape.setter
    def shape(self, shape: Tuple) -> None:
        if shape == self._shape:
            return
        self._coordinates = make_coordinates(shape, flatten=False)
        self._shape = shape

    @property
    def coordinates(self) -> np.ndarray:
        '''Coordinates of pixels in image data'''
        return self._coordinates

    @property
    def data(self) -> np.ndarray:
        '''image data'''
        return self._data

    @data.setter
    def data(self, data: Optional[np.ndarray]) -> None:
        if data is None:
            data = np.ndarray([])
        self._features = []
        self._results = pd.DataFrame()
        self.shape = data.shape
        self._data = data

    @property
    def results(self) -> pd.DataFrame:
        '''DataFrame containing tracking and characterization results'''
        return self._results

    @property
    def features(self) -> List[Feature]:
        '''List of objects of type Feature'''
        return self._features

    def detect(self) -> AnyFrame:
        '''Detect and localize features in data
        '''
        self._results = self.localizer.detect(self.data)
        self._features = []
        for _, feature in self._results.iterrows():
            (x0, y0), w, h = feature.bbox
            dim = min(w, h)
            d = self.data[y0:y0+dim, x0:x0+dim]
            c = self.coordinates[:, y0:y0+dim, x0:x0+dim].reshape((2, -1))
            this = Feature(data=d, coordinates=c)
            this.properties = self.properties
            this.particle.x_p = feature.x_p
            this.particle.y_p = feature.y_p
            self._features.append(this)
        return self

    def estimate(self) -> AnyFrame:
        '''
        Estimate parameters for current features
        '''
        for feature in self.features:
            feature.estimate()
        return self

    def optimize(self) -> pd.DataFrame:
        '''
        Optimize adjustable parameters
        '''
        results = [feature.optimize() for feature in self.features]
        self._results = pd.DataFrame(results)
        return self._results

    def analyze(self,
                data: Optional[np.ndarray] = None) -> pd.DataFrame:
        '''
        Detect features, estimate parameters, and fit

        Parameters
        ----------
        data: numpy.ndarray
            Normalized holographic microscopy data.

        Returns
        -------
        results: pandas.DataFrame
            Optimized parameters of generative model for each feature
        '''
        self.data = data
        return self.detect().estimate().optimize()


def example():
    from pathlib import Path
    import cv2

    basedir = Path(__file__).parent.parent.resolve()
    filename = str(basedir / 'docs' / 'tutorials' / 'image0010.png')
    data = cv2.imread(filename, cv2.IMREAD_GRAYSCALE).astype(float)
    frame = Frame()
    frame.data = data/100.
    frame.detect().estimate()
    print(frame.features)


if __name__ == '__main__':
    example()
