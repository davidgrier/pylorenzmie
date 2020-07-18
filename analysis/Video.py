#!/usr/bin/env python
# -*- coding: utf-8 -*-

import trackpy as tp
import pandas as pd
import json
import os
from .Frame import Frame
from .Trajectory import Trajectory


class Video(object):

    def __init__(self, frames=[], video_path=None, instrument=None, fps=None, info=None):
        self._fps = None
        self._frames = []
        self._instrument = instrument
        self.video_path = video_path
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
    
    def add(self, frame, framenum=None):
        if frame is None:
            return
        elif isinstance(frame, Frame):
            if frame.framenumber is not None:
                frame.framenumber = framenum
            self._frames.append(frame)
        elif isinstance(frame, np.ndarray) or isinstance(frame, String):
            self._frames.append(Frame(instrument=self.instrument, framenumber=framenum, image=frame))
        elif isinstance(frame, list):
            framenum = framenum if isinstance(framenum, list) else [None for f in frame]
            for i, _frame in frame:
                self.add(_frame, framenum[i])
        else:
            print('Warning: could not add frame of type {}'.format(type(Frame)))
                        
    @property 
    def path(self):
        return self._path
   
    @property 
    def filename(self):
        return self._filename 
            
    @property
    def video_path(self):
        if self.path is None or self.filename is None:
            return None
        else:
            return self.path + '/videos/' + self.filename + '.avi'
    
    @property 
    def image_path(self, framenum=None):
        if self.path is None or self.filename is None:
            return None
        else:
            image_path = self.path + '/' + self.filename + '_norm_images/'
            if framenum is not None:
                image_path += 'image' + str(framenum).rjust(4, '0') + '.png'
            return image_path 
    
    @video_path.setter
    def video_path.setter(self, path):
        if not isinstance(path, string):
            if path is not None:
                print('Warning: could not recognize path of type {}'.format(type(path)))
            self._path = None
            self._filename = None
        elif len(path.split('videos/')) is 2:
            self._path, self._filename = path.split('videos/')
            if self._filename[-4:] is '.avi':
                self._filename = self._filename[:-4]
        else:
            pathlist = path.split('/')
            if len(pathlist) is 1:
                print('Warning: path {} is incomplete'.format(path))
                self.video_path = None
            else:
                self._filename = pathlist[-1]
                self._path = '/'.join(pathlist[:-1])
        
        if self.video_path is not None:
            self.add( [Frame(image_path=path) for path in os.listdir(self.image_path())] )
            #### TODO: load in predictions from MLpreds/refined using regexpressions
            
            
        
    
        
                                      
               
           
       
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
                
            
