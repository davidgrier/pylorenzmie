from pylorenzmie.analysis import (Localizer, Feature)
from pylorenzmie.lib import LMObject
from pylorenzmie.utilities import coordinates as make_coordinates
import pandas as pd
import numpy as np
from typing import (Optional, Tuple, List, Dict)


class Frame(LMObject):
    '''
    Abstraction of a holographic microscopy video frame.
    ...

    Properties
    ----------
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
    detect() : int
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
                 data: Optional[np.ndarray] = None,
                 **kwargs) -> None:
        self._shape = (0, 0)
        self._data = data
        self.localizer = Localizer(**kwargs)
        self.kwargs = kwargs

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

    def detect(self) -> Frame:
        '''Detect and localize features in data
        '''
        self._results = self.localizer.detect(self.data)
        self._features = []
        for _, feature in self._results.iterrows():
            (x0, y0), w, h = feature.bbox
            dim = min(w, h)
            d = self.data[y0:y0+dim, x0:x0+dim]
            c = self.coordinates[:, y0:y0+dim, x0:x0+dim].reshape((2, -1))
            this = Feature(data=d, coordinates=c, **self.kwargs)
            this.particle.x_p = feature.x_p
            this.particle.y_p = feature.y_p
            self._features.append(this)
        return self

    def estimate(self) -> Frame:
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
