#!/usr/bin/env python
# -*- coding: utf-8 -*-

import trackpy as tp
import pandas as pd
import json
import os
from .Frame import Frame


#### path format: a video at home/experiment1/videos/myrun.avi has path 'home/experiment1/' (don't forget '/' at the end!) and filename 'myrun'. If your working directory is already the main directory (i.e. this file is  in experiment1) then path can be blank.
#### Path setters are designed so that the full video path (path='home/experiment1/videos/myrun.avi') and no filename will give the same result
class Video(object):

    def __init__(self, frames={}, path=None, video_path=None, instrument=None, fps=30, info=None):
        self._frames = {}
        self._fps = None
        self._instrument = instrument
        self._path = None
        self._video_path = None
        self.path = path
        if self.video_path is None: self.video_path = video_path  
        if len(frames) > 0: self.add(frames)
#         self._trajectories = []
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
        return list(self._frames.values())
    
    @property
    def framenumbers(self):
        return list(self._frames.keys())
    
    def get_frame(self, framenumber):
        return self._frames[framenumber]

    def get_frames(self, framenumbers):
        return [self.get_frame(fnum) for fnum in framenumbers]
    
    def set_frame(self, frame=None, framenumber=None):
        if frame is None:
            frame=Frame(framenumber=framenumber, path=self.path)
        if framenumber is not None:
            frame.framenumber = framenumber
        if frame.framenumber is None:
            print('Cannot set frame without framenumber')
        else:
            if frame.framenumber in self.framenumbers:
                print('Warning - overwriting frame {}'.format(framenumber))
            self._frames[frame.framenumber] = frame
            frame.path = self.path
            if self.instrument is not None:
                frame.instrument = iself.instrument
    
    def set_frames(self, frames=None, framenumbers=None):
        if frames is None and framenumbers is None: 
            return
        elif isinstance(frames, dict):
            self.set_frames(frames=list(frames.values()), framenumbers=list(frames.keys()))
            return
        elif framenumbers is None:
            framenumbers = [None for frame in frames]
        elif frames is None:
            frames = [Frame(path=self.path) for fnum in framenumbers]
        for i in range(len(frames)):
            self.set_frame(frames[i], framenumbers[i])
                           
    def add(self, frames):
        print('Adding {} frames to the end of video...'.format(len(frames)))
        if len(self.frames) > 0:
            nframes = max(self.framenumbers()) + 1
        else:
            nframes = 0
        self.set_frames( frames=frames, framenumbers = list(range(nframes, nframes+len(frames))) )
             
    def sort(self):
        self._frames = dict(sorted(self._frames.items(), key=lambda x: x[0]))        
            
    def clear(self):
        self._frames = {}
        
        
    @property 
    def video_path(self):
        return self._video_path        

    @video_path.setter                   
    def video_path(self, path):
        if not isinstance(path, str):
            self._video_path = None
        elif len(path) < 4 or path[-4:] !='.avi':
            print('error: video path must lead to file of type .avi')
        else:
            self._video_path = path
            if self.path is None:
                path = path.replace('videos/', '')[:-4]
                print('obtained path {} from corresponding video path'.format(path))
                self.path = path
           
    @property
    def path(self):
        return self._path
    
    @path.setter
    def path(self, path):
        if not isinstance(path, str):
            self._path = None
        elif len(path) >= 4 and path[-4:] == '.avi':
            self.video_path = path
        elif '.' in path:
            print('warning - {} is an invalid directory name'.format(path))
            self._path = None
        else:
            self._path = path
            if os.path.isdir(path):
#                 print('set path to existing directory {}'.format(path))
                if self.video_path is None:
                    if path[-1] == '/':
                        path = path[:-1]
                    filename = path.split('/')[-1]
#                     print(path.replace(filename, 'videos/'+filename) + '.avi')
#                     print(os.path.exists(path.replace(filename, 'videos/'+filename) + '.avi'))
                    
#                     print(path+'.avi')
                    if os.path.exists(path.replace(filename, 'videos/'+filename) + '.avi'):
                        self.video_path = path.replace(filename, 'videos/'+filename) + '.avi'
                    elif os.path.exists(path + '.avi'):
                        self.video_path = path + '.avi'

            else:
                print('setting path to new directory at path {}'.format(path))
                os.mkdir(path)
        
    def load(self):
        if self.path is not None:
            self.set_frames(frames=[Frame(path=self.path + '/norm_images/' + s) for s in os.listdir(self.path + '/norm_images')])
                
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
                
            
