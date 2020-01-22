#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import trackpy as tp
import pandas as pd
import json
from pylorenzmie.theory.Frame import Frame
from pylorenzmie.theory.Trajectory import Trajectory


class Video(object):

    def __init__(self, frames=[], info=None):
        self._frames = frames
        self._trajectories = []
        self.set_trajectories()
        self.deserialize(info)

    @property
    def frames(self):
        return self._frames

    def add(self, frame):
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
                self.add(Frame(info=d))
