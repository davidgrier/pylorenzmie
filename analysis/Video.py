#!/usr/bin/env python
# -*- coding: utf-8 -*-

import trackpy as tp
import pandas as pd
import json
import os
from .Frame import Frame
from .Trajectory import Trajectory


#### path format: a video at home/experiment1/videos/myrun.avi has path 'home/experiment1/' (don't forget '/' at the end!) and filename 'myrun'. If your working directory is already the main directory (i.e. this file is  in experiment1) then path can be blank.
#### Path setters are designed so that the full video path (path='home/experiment1/videos/myrun.avi') and no filename will give the same result
class Video(object):

    def __init__(self, frames=[], path=None, filename=None, instrument=None, fps=30, info=None):
        self._frames = []
        self._fps = None
        self._instrument = instrument
        self.path = path
        self.filename = filename
        self.add(frames)
        self._trajectories = []
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
    
    def add(self, frames, framenums=[]):
        if frames is None:
            return
        frames = [frames] if not isinstance(frames, list) else frames                               #### Ensure input is a list
        framenums = [None for frame in frames] if len(framenums) != len(frames) else framenums      #### Ensure framenums is same size as frames
        for i, frame in enumerate(frames):    
            if isinstance(frame, Frame):
                if framenums[i] is not None:
                    frame.framenumber = framenums[i]
                self._frames.append(frame)
            elif isinstance(frame, str):
                self._frames.append(Frame(instrument=self.instrument, framenumber=framenums[i], image_path=frame))
            elif isinstance(frame, np.ndarray): 
                self._frames.append(Frame(instrument=self.instrument, framenumber=framenums[i], image=frame))
            elif frame is not None:
                print('Warning: could not add frame of type {}'.format(type(Frame)))
            
    def sort(self):
        self._frames = sorted(self._frames, key=lambda x: x.framenumber if x.framenumber is not None else 100000)        
    
    def renumber(self):
        for i, frame in enumerate(self._frames):
            frame.framenumber = i
            
    def clear(self):
        self._frames = []
        
    @property 
    def path(self):
        return self._path
    
    @path.setter                   
    def path(self, path):
        self._path = path if isinstance(path, str) else ''    #### Ensure path is always string type
    
    @property 
    def filename(self):
        return self._filename 
            
    @filename.setter                                          #### Update filename and search for video
    def filename(self, filename):             
        if isinstance(filename, str):                         #### If filename is a string, remove suffix (if present) and get frames
            self._filename = filename.split('.avi')[0]
            if os.path.exists(self.images_path):              #### If we have a valid im directory, read frames one-at-a-time
                frames = [Frame(image_path=self.images_path+name) for name in os.listdir(self.images_path)] 
                self.add(frames)
            elif os.path.exists(self.video_path):
                self.get_normalized_video()       
        else:                                                   #### If invalid or None filename is passed, look for path+filename in self.path
            self._filename = None               
            if len(self.path.split('videos/')) is 2:        
                path, filename = self.path.split('videos/')
                self.path = path
                self.filename = filename                        #### If filename found in self.path, call the setter again
            elif filename is not None:
                print('Warning: invalid filename of type {}'.format(type(filename)))
 
    @property
    def video_path(self):
        return None if self.filename is None else self.path + 'videos' + self.filename + '.avi'
    
    @property 
    def images_path(self):
        return None if self.filename is None else self.path + self.filename + '_norm_images/' 
    
    
    def get_normalized_video(self):
        pass     #### background path is self.path+'videos/background.avi'
    
    @property
    def trajectories(self):
        return self._trajectories

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
        if 'video_path' in info.keys():
            if info['video_path'] is not None:
                self.video_path = info['video_path']
        if 'video_name' in info.keys():
            if info['video_path'] is not None:
                self.video_path = info['video_name']
                
            
