# -*- coding: utf-8 -*-

import numpy as np
from pylorenzmie.theory.Instrument import Instrument


class Frame(object):

    def __init__(self, instrument=None, **kwargs):
        self._detector = None  # should be YOLO
        self._features = []
        if instrument is None:
            self.instrument = Instrument(**kwargs)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        if data is None:
            return
        self._data = np.float(data)
        self._features = self._detector.predict(self._data)

    @property
    def features(self):
        return self._features
