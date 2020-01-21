#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import trackpy as tp
import pandas as pd
import json
from pylorenzmie.theory.LMHologram import LMHologram
from pylorenzmie.theory.Frame import Frame
from pylorenzmie.theory.Trajectory import Trajectory


class Video(object):

    def __init__(
            self, frames=[[]], images=[], save_empty=True, info=None):
        self.model = LMHologram()
        if info is not None:
            self._frames = []
            for i, frame in enumerate(frames):
                if not save_empty and len(frame[i]) == 0:
                    continue
                self.appendFrame(
                    images[i], frame[i], model=self.model, frame_no=i)
            self._trajectories = None
            self.setTrajectories()
        else:
            self.deserialize(info)

    @property
    def frames(self):
        return self._frames

    @property
    def trajectories(self):
        return self._trajectories

    def setTrajectories(self):
        pass

    def appendFrame(self, image, features, **kwargs):
        self._frames.append(
            Frame(data=image, features=features, **kwargs))

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
                info.pop()
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
                self.append(Trajectory(model=self.model, info=d))
        elif 'frames' in info.keys():
            for d in info['frames']:
                self.append(Frame(model=self.model, info=d))
