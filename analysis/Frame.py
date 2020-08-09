#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2, json
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from .Feature import Feature

class Frame(object):
    '''
    Abstraction of an experimental video frame. 
    ...
    Attributes
    ----------
    features : List of Feature objects
    bboxes : List of tuples ( {x, y, w, h} )
        Bounding box of dimensions (w, h) around feature at (x, y). Used for cropping to obtain image stamps
    framenumber : int
    path : string
        path leading to a base directory with data related to this particular experiment.
    image_path : string    
        path leading to the frame's corresponding .png image. By default, this is path/norm_images/image####.png (where #### is the framenumber)     
    image_path : string               
        path leading to the .png image file for this particular video frame. 
     ** Note: image_path doesn't have a setter - both path and image_path are determined by the framenumber and the path setter.
          If path setter gets a filename (i.e. frame.path='exp/image0123.png' or frame.path='myexp/norm_images/image0123.png') 
              then it sets the image_path; the path is set to the directory 'myexp'; and if framenumber isn't set, it's obtained
              from the filename (in this case, framenumber=123)
          If the path setter gets a directory (i.e. vid.path='myexp') then it sets the path and, if framenumber is set, checks 
              for a corresponding image ('myexp/image0123.png' or 'myexp/norm_images/image0123.png') and sets the image_path 
              (if file is found). 
    
    image : numpy.ndarray
        Image from camera. If None (default), the getter tries to read from local image_path. To save image, call load()
    
    instrument : Instrument
        Instrument instance used for prediction
        
    Methods ##TODO
    ------- 
    add(features=[], bboxes=[]):
        
    add(features, info=None) 
        features: list of Feature objects / list of serialized Features (dicts) / or list of bboxes (tuples)
        optional: info: (list of) serialized feature info (dicts)
            - Unpack and add each Feature to the Frame, 
            - if each input is a bbox, add it to Frame; otherwise, add None to frame's bbox list
            - deserialize info into Features, if info passed
    
    remove(index)
        index : list of integers. Remove features and bboxes at indices.
        
    load() 
        read image from local image_path 
        
    '''
    
    def __init__(self, features=None, instrument=None, 
                 framenumber=None, image=None, path=None, info=None):
        self._instrument = instrument
        self._framenumber = framenumber
        self._image = None
        self._image_path = None
        self._path = None
        self.image = image
        self.path = path
        self._bboxes = []
        self._features = []
        self.add(features=features)
        self.deserialize(info)
    
    @property
    def instrument(self):
        return self._instrument

    @instrument.setter
    def instrument(self, instrument):
        self._instrument = instrument
        for feature in self.features:
            if feature.model is not None:
                feature.model.instrument = instrument
    
    @property
    def framenumber(self):
        return self._framenumber

    @framenumber.setter
    def framenumber(self, idx):
        self._framenumber = idx
        self.path = self.path
#         if self.image_path is None and self.path is not None:
#             self.path = self.path
       
    
    @property
    def path(self):
        return self._path
    
    @path.setter
    def path(self, path):
        if not isinstance(path, str):               
            self._path = None
        elif len(path) >= 4 and path[-4:] == '.png':  #### If path is a file, then set image_path 
            self._image_path = path        
            filename = path.split('/')[-1]
            if self.framenumber is None and len(filename) > 8:
                try:                        #### Try to read framenumber from end of path (i.e. 0107 from '...image0107.png')
                    self.framenumber = int(filename[-8:-4]) 
                except ValueError:
                    print('Warning - could not read integer framenumber from pathname')
            path = path.replace(filename, '')         #### remove filename (and norm_images/) to get base directory
            path = path.replace('norm_images/', '')
#             print('Set path to {}'.format(path))
            self._path = path
        elif '.' in path:    
            print('warning - {} is an invalid directory name'.format(path))
            self._path = None        
        else:
            self._path = path               #### If path is a directory, set path and use framenumber to search for image_path                       
            if os.path.isdir(path):       
#                 print('set path to existing directory {}'.format(path))
                if self.image_path is None and self.framenumber is not None:
                    if path[-1] !='/':          
                        path += '/'
                    filename = 'image' + str(self.framenumber).rjust(4, '0') + '.png'
                    if os.path.exists(path + filename):
                        self._image_path = path + filename
                    elif os.path.exists(path + 'norm_images/' + filename):
                        self._image_path = path + 'norm_images/' + filename

            else:         #### If path does not exist, make new directory
#                 print('setting path to new directory at path {}'.format(path))
                os.mkdir(path)      
   
    @property
    def image_path(self):
        return self._image_path
    
    @property
    def image(self):
        return self._image if self._image is not None else cv2.imread(self.image_path)
                
    def load(self):
        self._image = self.image
                    
    def unload(self):
        self._image = None   
      
    @image.setter    #### Warning: images set directly will be lost when they are unloaded() 
    def image(self, image):
        if isinstance(image, np.ndarray):
            print('Warning - image passed directly to Frame without use of a path!')
            self._image = image
 
   
    @property
    def features(self):
        return self._features
    
    @property
    def bboxes(self):
        return self._bboxes
    
    def add(self, features=[], bboxes=[]):
        features = [features] if not isinstance(features, list) else features           #### Ensure input is a list
        bboxes = [bboxes] if not isinstance(bboxes, list) else bboxes
        for i in range( max(len(bboxes), len(features)) ):
            bbox = bboxes[i] if i < len(bboxes) else None                               #### If no bbox/feature, use None
            feature = features[i] if i < len(features) else None                        
            if isinstance(feature, dict):                                               #### If feature is serialized, then deserialize
                feature = Feature(info=feature)     
            if not isinstance(feature, Feature) and bbox is not None:                   #### If bbox but no feature, pass empty feature
                if len(bbox) == 4:
                    feature = Feature()
            if isinstance(feature, Feature):                                            #### If we have a feature to add, set instrument and add with appropriate bbox
                if feature.model is not None and self.instrument is not None: 
                    feature.model.instrument = self.instrument
                self._features.append(feature)
                self._bboxes.append(bbox)
            elif feature is not None:
                print('Warning - could not add feature {} of type {}'.format(i, type(feature)))
#                 msg = "features must be list of Features"
#                 msg += " or deserializable Features"
#                 raise(TypeError(msg))
    
    def remove(self, index):
        if isinstance(index, int):
#             print(index)
            self._features.pop(index)
            bbox = self.bboxes.pop(index)
#             print(bbox)
        elif isinstance(index, Feature):
            try:
                self.remove(self.features.index(index))
            except ValueError:
                print('Warning - could not remove Feature: Feature not found')
                return
        elif isinstance(index, list):
            for i in sorted(list(index), reverse=True): 
                self.remove(i)

    def no_edges(self, tol=200, image_shape=(1280,1024)):
        image_shape = image_shape or np.shape(self.image)
        minwidth = np.min(image_shape)
        if tol < 0 or tol > minwidth/2:
            print('Invalid tolerance for this frame size')
            return None
        xmin, ymin = (tol, tol)
        xmax, ymax = np.subtract(image_shape, (tol, tol))
        
        toss = []
        for i, bbox in enumerate(self.bboxes):
            if bbox is not None and (bbox[0]<xmin or bbox[0]>xmax or bbox[1]<ymin or bbox[1]>ymax):
                toss.append(i)
        self.remove(toss)
    
    def nodoubles(self, tol=5):
        toss = []
        for i, bbox1 in enumerate(self.bboxes):
            for j, bbox2 in enumerate(self.bboxes[:i]):
                x1, y1 = bbox1[:2]
                x2, y2 = bbox2[:2]
                dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                if dist<tol:
                    toss.append(i)
                    break
        self.remove(toss)
        
    def optimize(self, report=True, **kwargs):
        for idx, feature in enumerate(self.features):
            result = feature.optimize(**kwargs)
            if report:
                print(result)

    def serialize(self, save=False, path='frames', omit=[], omit_feat=[]):
        info = {}
        if 'features' not in omit:
            info['features'] = [feature.serialize( exclude=omit_feat ) for feature in self.features]
        if 'bboxes' not in omit:
            info['bboxes'] = self.bboxes
        if self.image_path is not None:
            info['image_path'] = self.image_path
        if self.framenumber is not None:
            info['framenumber'] = str(self.framenumber)
        if save:
            path = path if self.path is None else self.path + '/' + path
            if not (len(path) >= 5 and path[-5:] == '.json'):
                if not os.path.exists(path): os.mkdir(path)
                framenumber = str(self.framenumber).rjust(4, '0') if self.framenumber is not None else ''
                path = path + '/frame{}.json'.format(framenumber)
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
        if 'framenumber' in info.keys():
            self.framenumber = int(info['framenumber']) 
        if 'image_path' in info.keys():                   
            self.path = info['image_path']                           
        features = info['features'] if 'features' in info.keys() else []
        bboxes = info['bboxes'] if 'bboxes' in info.keys() else []
        self.add(features=features, bboxes=bboxes) 
        
    def to_df(self, info=None):
        info = info if info is not None else self.serialize(omit_feat=['data'])
        features = info['features'] if 'features' in info.keys() else {}
        bboxes = info['bboxes'] if 'bboxes' in info.keys() else []
        df = pd.DataFrame.from_dict(features)
        df['bboxes'] = bboxes
        df['framenumber'] = info['framenumber'] if 'framenumber' in info.keys() else None  
        return df

    def show(self):
        image = self.image
        if image is None: return
        fig, ax = plt.subplots()
        ax.imshow(image, cmap='gray')
        for bbox in self.bboxes:
            if bbox is not None:
                x,y,w,h= bbox
                test_rect = Rectangle(xy=(x - w/2, y - h/2), width=w, height=h, fill=False, linewidth=3, edgecolor='r')
                ax.add_patch(test_rect)
        plt.show()
    
