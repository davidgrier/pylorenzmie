#!/usr/bin/env python
# -*- coding: utf-8 -*-

import trackpy as tp
import pandas as pd
import json
import os
from .Frame import Frame
from CNNLorenzMie.experiments.normalize_image import normalize_video
from CNNLorenzMIe.experiments.running_normal import running_normalize



class Video(object):

    '''
    Abstraction of an experimental video. 
    ...
    Attributes
    ----------
    frames : list
        list of the video's Frames.   
    framenumbers : list
        list of the each Frame's corresponding framenumber
     **Note: frames and framenumbers are stored as a dict as {framenumber: frame}, but the getters each return a list
    
    trajectories : DataFrame
        DataFrame containing information about each feature's properties, framenumber, and trajectory (if set)
    path : string
        path leading to a base directory with data related to this particular experiment.
    video_path : string               
        path leading to the .avi video file for this particular experiment. 
         ** Note: If the path is already given, the video_path can be determined using setDefaultPath(); and vice versa. (See below)
 ###        ** Note: If the path+framenumbers are already given, the individual image_paths can be determined using setDefaultImagePaths(); and vice versa. (See below)
    
        ** Note: video_path does not have a setter - both path and video_path are determined by the path setter.
          If the path setter gets a filename (i.e. vid.path='folder/myexp.avi' or vid.path='folder/videos/myexp.avi') then
              it sets the video_path, and the path is set to the directory 'folder/myexp'
          If the path setter gets a directory (i.e. vid.path='folder/myexp') then it sets the path and checks for a
              corresponding video ('folder/myexp.avi' or 'folder/videos/myexp.avi') and sets the video_path (if file is found).
              
    fps : float
        camera frames/second
    instrument : Instrument
        Instrument instance used for prediction
   
    Methods
    ------- 
    set_frames(frames=None, framenumbers=None)
        frames : list of Frames  |  framenumbers : list of integers
        Set the video's frames. Ensures each frame has a framenumber.
         - If ONLY FRAMES are passed, the framenumbers are obtained from the frames via frame.framenumber
         - If FRAMES AND FRAMENUMBERS are BOTH passed, the passed framenumbers are used (i.e. frames[i].framenumber = framenumbers[i]
         - If ONLY FRAMENUMBERS are passed, frames are obtained via 'path' by searching for images at 'path/norm_images/image####.png'
         - If NEITHER are passed, frames and framenumbers obtained by reading all of the files in 'path/norm_images'

    get_frames(framenumbers) : list of frames
        Return frames indexed by corresponding framenumbers
    add(frames):
        Add a list of frames to the end of the video, with framenumber starting at the maximum current framenumber
    sort():
        Sort the list of frames by framenumber
    clear:
        Clear current frames
        
    set_trajectories(link=True, **kwargs)
        Set trajectories by looping over Frames and Features and storing Feature properties into a DataFrame.
        if Link=True, link the trajectories using trackpy and any relevant trackpy **kwargs.
    clear_trajectories()
        Set trajectories to an empty DataFrame
        
    serialize(save=False, path='', traj_path=None, omit=[], omit_frame=[], omit_feat=[])
        Convert Video object into a dict containing frames (dict to serialized frames), fps, and video_path.
         - If save=True, save to 'self.path/path' (or 'self.path/path/video.json' if path doesn't end in '.json')
         
    deserialize(info):
        Read information from dict into Video
    '''
    
    def __init__(self, frames={}, path=None, instrument=None, fps=30, info=None):
        self._frames = {}
        self.fps = fps
        self.instrument = instrument
        self.path = None
        self.video_path = None
        self.bg_path = None
        self.setDefaultPath(path)
        self._trajectories = pd.DataFrame()
        if len(frames) > 0: self.add(frames)
        self.deserialize(info)

    @property
    def instrument(self):
        return self._instrument

    @instrument.setter
    def instrument(self, instrument):
        self._instrument = instrument
        
    @property
    def fps(self):
        return self._fps

    @fps.setter
    def fps(self, fps):
        self._fps = float(fps)    
        
    @property
    def frames(self):
        return list(self._frames.values())
    
    @property
    def framenumbers(self):
        return list(self._frames.keys())

    @property
    def trajectories(self):
        return self._trajectories 

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
            self._frames[frame.framenumber] = frame
            frame.path = self.path
            if self.instrument is not None:
                frame.instrument = self.instrument
        return frame
    
    def set_frames(self, frames=None, framenumbers=None):
        if frames is None and framenumbers is None: 
            if self.path is not None:
                print('Setting frames using contents of path {}/norm_images:'.format(self.path))
                self.set_frames(frames=[Frame(path=self.path + '/norm_images/' + s) for s in os.listdir(self.path + '/norm_images')])
                self.sort()
                return
        elif isinstance(frames, dict):
             self._frames.update(frames)
             
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
    
    def remove(self, framenumbers):
        for framenumber in framenumbers:
            if framenumber in self.framenumbers:
                del self._frames[framenumber]
                
    def sort(self):
        self._frames = dict(sorted(self._frames.items(), key=lambda x: x[0]))        
            
    def clear(self):
        self._frames = {}
                      
    def setDefaultPath(self, path=None, viddir='videos/'):
        if path is None:
            path = self.path or self.video_path
        if not isinstance(path, str):
            return
        
        if len(path) >= 4 and path[-4:] == '.avi':
            self.video_path = path
            self.bg_path = self.video_path.replace(self.video_path.split('/')[-1], 'background.avi')
            self.path = path.replace(viddir, '')[:-4]
        else:
            if '.' in path:
                print('warning - {} is an invalid directory name'.format(path))
                return
            self.path = path
            if not os.path.isdir(path):
                os.mkdir(path)
            if self.video_path is None:
                if path[-1] == '/':
                    path = path[:-1]
                filename = path.split('/')[-1]
                base_path = path.replace(filename, '')
                base_path = base_path.replace(viddir, '')
                self.video_path = '/'.join([base_path, viddir, filename + '.avi'])

    def set_trajectories(self, link=True, search_range=2., verbose=True, **kwargs):
        df = self.trajectories
        if df.empty:
            for frame in self.frames:
                df = df.append(frame.to_df())
        if link:
            if not verbose:
                tp.quiet(suppress=True)
            # display(df)
            print(df)
            df = df.rename(columns={'x_p':'x', 'y_p':'y', 'framenumber':'frame'})
            # display(df)
            print(df)
            df = tp.link_df(df, search_range, **kwargs)
            df = df.rename(columns={'x':'x_p', 'y':'y_p', 'frame':'framenumber'})
        self._trajectories = df
     
    def clear_trajectories(self):
        self._trajectories = pd.DataFrame()

    def normalize(self, save_folder=None, order=2, dark=None):
        if save_folder is None:
            save_folder = self.path + '/norm_images/'
        else:
            print('not sure how to handle this yet')
        if self.bg_path is None: #assume if no background video that this is a flow experiment
            running_normalize(self.video_path, save_folder = save_folder, order = order, dark = dark)
        else: #use background video if there is one
            normalize_video(self.bg_path, self.video_path, save_folder = save_folder, order = order)
        self.set_frames()
            
    def serialize(self, save=False, framenumbers=None, path=None, traj_path=None,
                  omit=[], omit_frame=[], omit_feat=[]):
        info={}
        if traj_path is not None:
            self.trajectories.to_csv(traj_path)
            info['trajectories'] = traj_path    
        if 'frames' not in omit:
            framenumbers = framenumbers or self.framenumbers
            frames = [self._frames[key].serialize(omit=omit_frame, omit_feat=omit_feat) for key in framenumbers]
            info['frames'] = dict(zip(framenumbers, frames))
        info['fps'] = self.fps
        if self.path is not None:
            info['path'] = self.path
        if self.video_path is not None:
            info['video_path'] = self.video_path
        if save:
            path = path or self.path
            if path is None:
                return info
            if not (len(path) >= 5 and path[-5:] == '.json'):
                if not os.path.exists(path): os.mkdir(path)
                path += '/video.json'
                path = path.replace('//', '/')
            with open(path, 'w') as f:
                json.dump(info, f)
        return info

    def deserialize(self, info):
        if info is None:
            return
        if isinstance(info, str):
            with open(info, 'rb') as f:
                info = json.load(f)
        if 'trajectories' in info.keys():
            traj_path = info['trajectories']
            self._trajectories = traj_path
#             self._trajectories = trajs
        if 'frames' in info.keys():
            framenumbers = list(info['frames'].keys())
            frames = [Frame(info=d) for d in info['frames'].values()]
            self.set_frames(framenumbers=framenumbers, frames=frames)
        if 'fps' in info.keys():
            if info['fps'] is not None:
                self.fps = float(info['fps'])
        if 'path' in info.keys():
            if info['path'] is not None:
                self.path = info['path']
        if 'video_path' in info.keys():
            if info['video_path'] is not None:
                self.video_path = info['video_path']
       
            
