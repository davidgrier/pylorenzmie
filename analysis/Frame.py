import cv2
import numpy as np
import pandas as pd
from .Feature import Feature
from pylorenzmie.utilities import coordinates
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
    bboxes : list
        List of tuples ((x, y), w, h)
        Bounding box of dimensions (w, h) around feature at (x, y). 
        Used for cropping to obtain image stamps
    features : list
        List of Feature objects corresponding to bboxes

    Methods
    -------
    analyze(image) : 
        Identify features in image that are associated with
        particles and optimize the parameters of those features.
        Returns pandas.DataFrame of optimized results.
    optimize():
        Optimize the model parameters for each of the features
        associated with the bboxes in the currently loaded image.

    '''
    def __init__(self, image=None, bboxes=None, **kwargs):
        self._data = None
        self._shape = None
        self._coordinates = None
        self.localizer = Localizer(**kwargs)
        self.estimator = Estimator(**kwargs)
        self.image = image
        self._features = []
        self.bboxes = bboxes or []
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
            self._coordinates = coordinates(shape, flatten=False)
        self._shape = shape

    @property
    def coordinates(self):
        return self._coordinates
    
    @property
    def data(self):
        '''image data'''
        return self._data

    @data.setter
    def data(self, data):
        if data is None:
            self._data = None
        else:
            if data.shape != self.shape:
                self.shape = data.shape
        self._data = data

    @property
    def features(self):
        return self._features
    
    @property
    def bboxes(self):
        return self._bboxes

    @bboxes.setter
    def bboxes(self, bboxes):
        self._bboxes = bboxes
        self._features = []
        for bbox in bboxes:
            ((x0, y0), w, h) = bbox
            data = self.data[y0:y0+h, x0:x0+w]
            coordinates = self.coordinates[:, y0:y0+h, x0:x0+w]
            feature = Feature(data=data,
                              coordinates=coordinates.reshape((2,-1)),
                              **self.kwargs)
            self._features.append(feature)

    def analyze(self, data=None):
        '''
        Localize features, estimate parameters, and fit

        Parameters
        ----------
        data: numpy.ndarray
            Normalized holographic microscopy data.

        Returns
        -------
        results: pandas.DataFrame
            Optimized parameters of generative model for each feature
        '''
        if data is not None:
            self.data = data
        centers, bboxes = self.localizer.predict(self.data)
        self.bboxes = bboxes
        for feature, center in zip(self.features, centers):
            particle = feature.particle
            properties = self.estimator.predict()
            properties['x_p'] = center[0]
            properties['y_p'] = center[1]
            particle.properties = properties
        return self.optimize()

    def optimize(self):
        '''Optimize adjustable parameters'''
        results = [feature.optimize() for feature in self.features]
        return pd.DataFrame(results)
        
