import cv2
import numpy as np
import pandas as pd
from .Feature import Feature
from pylorenzmie.utilities import coordinates as make_coordinates
from pylorenzmie.detection import (Localizer, Estimator)


class Frame(object):
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
    bboxes : list
        List of bounding boxes: [(x0, y0), w, h] for each feature
        identified by detect() or analyze()
    results : pandas.DataFrame
        Summary of tracking and characterization data obtained by estimate(),
        optimize() or analyze()
 
    Methods
    -------
    detect() : int
        Detect and localize features in data. Sets features and bboxes

        Returns
        -------
        nfeatures : int
            Number of features detected

    estimate() :
        Estimate particle position and characteristics for each feature

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
    def __init__(self, **kwargs):
        self._data = None
        self._shape = None
        self._coordinates = None
        self.localizer = Localizer(**kwargs)
        self.estimator = Estimator(**kwargs)
        self._features = []
        self._bboxes = []
        self._results = None
        self.kwargs = kwargs
        
    @property
    def shape(self):
        '''image shape'''
        return self._shape

    @shape.setter
    def shape(self, shape):
        if shape == self._shape:
            return
        if shape is None:
            self._coordinates = None
        else:
            self._coordinates = make_coordinates(shape, flatten=False)
        self._shape = shape

    @property
    def coordinates(self):
        '''Coordinates of pixels in image data'''
        return self._coordinates
    
    @property
    def data(self):
        '''image data'''
        return self._data

    @data.setter
    def data(self, data):
        if data is not None:
            if data.shape != self.shape:
                self.shape = data.shape
        self._data = data

    @property
    def features(self):
        '''List of objects of type Feature'''
        return self._features

    @property
    def bboxes(self):
        '''List of bounding boxes'''
        return self._bboxes

    @property
    def results(self):
        '''DataFrame containing tracking and characterization results'''
        return self._results

    def detect(self):
        '''
        Detect and localize features in data
        '''
        discoveries = self.localizer.detect(self.data)
        self._bboxes = [discovery['bbox'] for discovery in discoveries]
        self._features = []
        for discovery in discoveries:
            ((x0, y0), w, h) = discovery['bbox']
            dim = min(w, h)
            data = self.data[y0:y0+dim, x0:x0+dim]
            coordinates = self.coordinates[:, y0:y0+dim, x0:x0+dim]
            feature = Feature(data=data,
                              coordinates=coordinates.reshape((2,-1)),
                              **self.kwargs)
            feature.particle.x_p = discovery['x_p']
            feature.particle.y_p = discovery['y_p']
            self._features.append(feature)
        return len(discoveries)

    def estimate(self):
        '''
        Estimate parameters for current features
        '''
        predict = self.estimator.predict
        for feature in self.features:
            feature.particle.properties = predict(feature.data)

    def optimize(self):
        '''
        Optimize adjustable parameters
        '''
        results = [feature.optimize() for feature in self.features]
        self._results = pd.DataFrame(results)
        return self._results

    def analyze(self, data=None):
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
        self.detect()
        self.estimate()
        return self.optimize()     
