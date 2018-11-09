#!/usr/bin/env python
# -*- coding: utf-8 -*-

from GeneralizedLorenzMie import GeneralizedLorenzMie
from LMHologram import LMHologram as Model


class Feature(object):
    '''Abstraction for a feature in an in-line hologram'''

    def __init__(self,
                 data=None,
                 coordinates=None):
        self.model = Model(coordinates=coordinates)
        self.coordinates = self.model.coordinates
        self.data = data

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        if isinstance(model, GeneralizedLorenzMie):
            self._model = model


if __name__ == '__main__':
    import numpy as np

    x = np.arange(4)
    r = np.stack((x, x, x))
    a = Feature(coordinates=r)
    print(a.shape)
