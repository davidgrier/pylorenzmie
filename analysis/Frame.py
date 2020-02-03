#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from .Feature import Feature


class Frame(object):

    def __init__(self, features=None, instrument=None,
                 framenumber=None, info=None):
        self._instrument = instrument
        self._framenumber = framenumber
        self._features = []
        if features is not None:
            for feature in features:
                if isinstance(feature, dict):
                    self.features.append(Feature(info=feature))
                elif type(feature) is Feature:
                    self.features.append(feature)
                else:
                    msg = "features must be list of Features"
                    msg += " or deserializable Features"
                    raise(TypeError(msg))
        if info is not None:
            self.deserialize(info)

    @property
    def instrument(self):
        return self._instrument

    @instrument.setter
    def instrument(self, instrument):
        self._instrument = instrument

    @property
    def framenumber(self):
        return self._framenumber

    @framenumber.setter
    def framenumber(self, idx):
        self._framenumber = idx

    @property
    def features(self):
        return self._features

    def add(self, features):
        for feature in features:
            if self.instrument is not None:
                feature.model.instrument = self.instrument
            self._features.append(feature)

    def optimize(self, report=True, **kwargs):
        for idx, feature in enumerate(self.features):
            result = feature.optimize(**kwargs)
            if report:
                print(result)

    def serialize(self, filename=None, omit=[], omit_feat=[]):
        info = {}
        features = []
        if 'features' not in omit:
            for feature in self.features:
                out = feature.serialize(exclude=omit_feat)
                features.append(out)
        info['features'] = features
        if self.framenumber is not None:
            info['framenumber'] = str(self.framenumber)
        for k in omit:
            if k in info.keys():
                info.pop(k)
        if filename is not None:
            with open(filename, 'w') as f:
                json.dump(info, f)
        return info

    def deserialize(self, info):
        if info is None:
            return
        if isinstance(info, str):
            with open(info, 'rb') as f:
                info = json.load(f)
        if 'features' in info.keys():
            features = info['features']
            self._features = []
            for d in features:
                self.features.append(Feature(info=d))
        if 'framenumber' in info.keys():
            self.framenumber = int(info['framenumber'])
        else:
            self.framenumber = None
