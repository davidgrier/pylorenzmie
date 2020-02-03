#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from .Feature import Feature


class Trajectory(object):

    def __init__(self, features=[], framenumbers=[],
                 info=None, instrument=None):
        self._features = []
        self._framenumbers = []
        self._instrument = instrument
        if features is not None:
            for idx, feature in enumerate(features):
                if isinstance(feature, dict):
                    f = Feature(info=feature)
                elif type(feature) is Feature:
                    f = feature
                self.add([f], [framenumbers[idx]])
        if info is not None:
            self.deserialize(info)

    @property
    def instrument(self):
        return self._instrument

    @instrument.setter
    def instrument(self, instrument):
        self._instrument = instrument

    @property
    def features(self):
        return self._features

    @property
    def framenumbers(self):
        return self._framenumbers

    def add(self, features, framenumbers):
        if len(features) != len(framenumbers):
            msg = "features and framenumbers must be same length."
            raise(ValueError(msg))
        for idx, feature in enumerate(features):
            if self.instrument is not None:
                feature.instrument = self.instrument
            self._features.append(feature)
            self._framenumbers.append(framenumbers[idx])

    def serialize(self, filename=None, omit=[], omit_feat=[]):
        features = []
        framenumbers = []
        for idx, feature in enumerate(self.features):
            if 'features' not in omit:
                out = feature.serialize(exclude=omit_feat)
                features.append(out)
            if 'framenumbers' not in omit:
                framenumbers.append(int(self.framenumbers[idx]))
        info = {'features': features,
                'framenumbers': framenumbers}
        for k in omit:
            if k in info.keys():
                info.pop()
        if filename is not None:
            with open(filename, 'w') as f:
                json.dump(out, f)
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
                self._features.append(Feature(info=d))
        if 'framenumbers' in info.keys():
            self._framenumbers = info['framenumbers']

    def optimize(self, report=True, **kwargs):
        for idx, feature in enumerate(self.features):
            result = feature.optimize(**kwargs)
            if report:
                print(result)
