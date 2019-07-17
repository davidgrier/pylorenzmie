# -*- coding: utf-8 -*-

import numpy as np
from pylorenzmie.theory.Instrument import Instrument
from pylorenzmie.theory.Frame import Frame


class Video(object):

    def __init__(self, images=[], instrument=None, **kwargs):
        self._detector = None  # should be YOLO
        self._estimator = None
        self._frames = []
        if instrument is None:
            self.instrument = Instrument(**kwargs)

    @property
    def frames(self):
        return self._frames
