import cv2
import numpy as np
import pandas as pd
from .Feature import Feature
from pylorenzmie.fitting import Optimizer
from pylorenzmie.utilities import coordinates
from pylorenzmie.detection import (Localizer, Estimator)


class Frame(object):
    '''
    Abstraction of a holographic microscopy video frame. 
    ...

    Properties
    ----------
    image : numpy.ndarray
        [w, h] array of image data
    data : numpy.ndarray
        automatically normalized version of image
    shape : tuple
        dimensions of image data, updated to reflect most recent image
    coordinates : numpy.ndarray
        [3, w, h] of pixel coordinates for most recent image
    dark_count : float or numpy.ndarray
        instrumental dark count of the camera
    background : float or numpy.ndarray
        background value or image for hologram normalization
    bboxes : list
        List of tuples ((x, y), w, h)
        Bounding box of dimensions (w, h) around feature at (x, y). 
        Used for cropping to obtain image stamps
    features : list
        List of Feature objects corresponding to bboxes

    Methods
    -------
    normalize(image) :
        Returns normalized version of image
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
        self.optimizer = Optimizer(**kwargs)
        self.image = image
        self._features = []
        self.bboxes = bboxes or []
        self.kwargs = kwargs

    @property
    def dark_count(self):
        return self.optimizer.model.instrument.dark_count

    @dark_count.setter
    def dark_count(self, dark_count):
        self.optimizer.model.instrument.dark_count = dark_count

    @property
    def background(self):
        return self.optimizer.model.instrument.background

    @background.setter
    def background(self, background):
        self.optimizer.model.instrument.background = background
        
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
    def image(self):
        '''image data'''
        return self._image

    @image.setter
    def image(self, image):
        if image is None:
            self._data = None
        else:
            if image.shape != self.shape:
                self.shape = image.shape
            self._data = self.normalize(image)
        self._image = image

    @property
    def data(self):
        '''normalized image data'''
        return self._data

    def normalize(self, image):
        return ((image - self.dark_count) /
                (self.background - self.dark_count))
        
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
            fitter = Feature(optimizer=self.optimizer, **self.kwargs)
            feature = dict(bbox=bbox,
                           data=data,
                           coordinates=coordinates,
                           fitter=fitter)
            self._features.append(feature)

    def analyze(self, image=None):
        '''
        Localize features, estimate parameters, and fit

        Parameters
        ----------
        image: numpy.ndarray
            Holographic microscopy data.

        Returns
        -------
        results: pandas.DataFrame
            Optimized parameters of generative model for each feature
        '''
        if image is not None:
            self.image = image
        centers, bboxes = self.localizer.predict(self.data)
        self.bboxes = bboxes
        for feature, center in zip(self.features, centers):
            particle = feature['fitter'].particle
            properties = self.estimator.predict()
            properties['x_p'] = center[0]
            properties['y_p'] = center[1]
            particle.properties = properties
        return self.optimize()

    def optimize(self):
        '''Optimize adjustable parameters'''
        results = []
        for feature in self.features:
            fitter = feature['fitter']
            fitter.data = feature['data'].ravel()
            fitter.coordinates = feature['coordinates'].reshape((2,-1))
            results.append(fitter.optimize())
        return pd.DataFrame(results)
        
