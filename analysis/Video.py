#!/usr/bin/env python
# -*- coding: utf-8 -*-

import trackpy as tp
import pandas as pd
import json
from .Frame import Frame
from .Trajectory import Trajectory


class Video(object):

    def __init__(self, frames=[], instrument=None, fps=None, info=None):
        self._fps = None
        self._frames = []
        self._instrument = instrument
        self.add(frames)
        self._trajectories = []
        self._traj_df = pd.DataFrame();
        self.deserialize(info)

    @property
    def fps(self):
        return self._fps

    @fps.setter
    def fps(self, fps):
        self._fps = float(fps)

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
            self._frames.append(frame)

    @property
    def trajectories(self):
        return self._trajectories
    @property
    def traj_df(self):
        return self._traj_df

    def set_trajectories(self, search_range=2., verbose=True, **kwargs):
        if not verbose:
            tp.quiet(suppress=True)
        d = {'x': [], 'y': [], 'frame': [], 'idx': []}
        for i, frame in enumerate(self.frames):
            for j, feature in enumerate(frame.features):
                d['x'].append(feature.model.particle.x_p)
                d['y'].append(feature.model.particle.y_p)
                d['idx'].append((i, j))
                d['frame'].append(frame.framenumber)
        df = tp.link_df(pd.DataFrame(data=d), search_range, **kwargs)
        self.traj_df = df;
        dfs = []
        if not pd.isnull(df.particle.max()):
            for particle in range(df.particle.max()+1):
                dfs.append(df[df.particle == particle])
        trajectories = []
        for idx in range(len(dfs)):
            features, framenumbers = ([], [])
            df = dfs[idx]
            for (i, j) in df.idx:
                features.append(self.frames[i].features[j])
                framenumbers.append(self.frames[i].framenumber)
            trajectories.append(Trajectory(instrument=self.instrument,
                                           features=features,
                                           framenumbers=framenumbers))
        self._trajectories = trajectories

    def serialize(self, filename=None,
                  omit=[], omit_frame=[], omit_traj=[], omit_feat=[]):
        trajs, frames = ([], [])
        if 'trajectories' not in omit:
            for traj in self.trajectories:
                trajs.append(
                    traj.serialize(omit=omit_traj, omit_feat=omit_feat))
        if 'frames' not in omit:
            for frame in self.frames:
                frames.append(
                    frame.serialize(omit=omit_frame, omit_feat=omit_feat))
        info = {'trajectories': trajs,
                'frames': frames}
        if self.fps is not None:
            info['fps'] = float(self.fps)
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
                self.trajectories.append(Trajectory(info=d))
        if 'frames' in info.keys():
            for d in info['frames']:
                self.add([Frame(info=d)])
        if 'fps' in info.keys():
            if info['fps'] is not None:
                self.fps = float(info['fps'])
