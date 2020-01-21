#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import json
from pylorenzmie.theory.Feature import Feature
from pylorenzmie.theory.LMHologram import LMHologram


class Trajectory(object):

    def __init__(self, features=None, info=None, model=None):
        if model is None:
            model = LMHologram()
        self._features = []
        if features is not None:
            for feature in features:
                if isinstance(feature, dict) or isinstance(feature, str):
                    f = Feature(model=self.model, info=feature)
                elif type(feature) is Feature:
                    f = feature
                self._features.append(f)
        if info is not None:
            self.deserialize(info)

    @property
    def features(self):
        return self._features

    @features.setter
    def features(self, features):
        self._features = features

    def serialize(self, filename=None, omit=[], **kwargs):
        features = []
        for feature in self.features:
            feature.data = None
            out = feature.serialize(**kwargs)
            features.append(out)
        info = {'features': features}
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
                self._features.append(Feature(model=self.model, info=d))

    def optimize(self, report=True, **kwargs):
        for idx, feature in enumerate(self.features):
            result = feature.optimize(**kwargs)
            if report:
                print(result)
