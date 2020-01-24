#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import trackpy as tp
import pandas as pd
import json
from pylorenzmie.theory import Frame, Trajectory, Instrument


class Video(object):

    def __init__(self, frames=[], instrument=None, info=None):
        self._frames = []
        self._instrument = instrument
        self.add(frames)
        self._trajectories = []
        self.set_trajectories()
        self.deserialize(info)

    @property
    def instrument(self):
        return self._instrument

    @instrument.setter
    def instrument(self, instrument):
        self._instrument = instrument

    @property
    def frames(self):
        return self._frames

    def add(self, frames):
        for frame in frames:
            if self.instrument is not None:
                frame.instrument = self.instrument
            self._frames.append(frame)

    @property
    def trajectories(self):
        return self._trajectories

    def set_trajectories(self):
        pass

    def serialize(self, filename=None,
                  omit=[], omit_frame=[], omit_traj=[], omit_feat=[]):
        trajs, frames = ([], [])
        for traj in self.trajectories:
            trajs.append(
                traj.serialize(omit=omit_traj, omit_feat=omit_feat))
        for frame in self.frames:
            frames.append(
                frame.serialize(omit=omit_frame, omit_feat=omit_feat))
        info = {'trajectories': trajs,
                'frames': frames}
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
        if 'trajectories' in info.keys():
            self._trajectories = []
            for d in info['trajectories']:
                self.add(Trajectory(info=d))
        elif 'frames' in info.keys():
            for d in info['frames']:
                self.add([Frame(info=d)])
