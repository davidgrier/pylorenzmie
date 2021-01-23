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

    Properties
    ----------
    features : list
        List of Feature objects
    bboxes : list
        List of tuples ( x, y, w, h )
        Bounding box of dimensions (w, h) around feature at (x, y). 
        Used for cropping to obtain image stamps
        FIXME: is (x,y) the center or the corner? It should be the corner.
    framenumber : int
    path : string
        path leading to a base directory with data related to this 
        particular experiment.
    image_path : string               
        path leading to the .png image file for this particular video frame. 
     ** Note: If the framenumber+path are already given, the image_path can be determined using setDefaultPath(); and vice versa. (See below)
    
    image : numpy.ndarray
        Image from camera. If None (default), the getter tries to read from local image_path. To store image, call load(); to write to image_path, call save()
    
    instrument : Instrument
        Instrument instance used for prediction. Setters ensure all of the Frame's Features share the same instrument.
        
        
    Methods
    ------- 
    add(features=[], bboxes=[]):
        features: list of Features (objects) or serialized Features (dicts) to be added
        bboxes: list of bboxes (tuples) to be added
        
        Add features and/or bounding boxes. Ensures that each Feature in frame.features corresponds to a bbox at the same index in frame.bboxes 
         -If only features (or more features than bboxes) are passed, then empty bboxes (i.e. None) are added to frame.bboxes instead
         -If only bboxes (or more bboxes than features) are passed, then empty features (i.e. Feature() instances) are added to frame.features instead
    
    remove(index)
        index : list of integers. Remove features and bboxes at indices.

        
    setDefaultPath(path=None, imdir='norm_images/')
        Set image_path and path/filename to default values, depending on the information given.
    
        path: String corresponding to the Frame's image_path (if string ends with '.png') or path (if string is a path to a directory)
         -If the input path ends in '.png', then it becomes the Frame's image_path; the Frame's path is obtained by removing the filename at the end (and the imdir, if present); 
          and if the framenumber is not set, it is obtained from the filename.
             For example, calling frame.setDefaultPath(path='myexp/norm_images/image0123.png') would set frame.framenumber=123 (if the framenumber wasn't already set),
             frame.image_path='myexp/norm_images/image0123.png', and frame.path='myexp/'
        -If the input path leads to a directory, the Frame's path is set; and if the framenumber is given, then the frame's image_path is set to path + 'image{framenumber}.png'
             For example, if frame already has frame.framenumber=123, then calling frame.setDefaultPath(path='myexp') would set the path to 'myexp' and 
             set the image_path to frame.image_path = 'myexp/norm_images/image0123/png'
        -If the frame obtains a path which is not an existing directory, a new folder is created
        -If no input path is given, the Frame's image_path will be used. If the frame.image_path is not set, frame.path is used; and if frame.path is also not set, the function does nothing
    
        imdir: String to set the image_path to a specified directory. 
             For example, if frame already has frame.framenumber=123, then calling frame.setDefaultPath(path='myexp', impath='myimages/') would set the path to 'myexp' and 
             set the image_path to frame.image_path='myexp/myimages/image0123/png'
    
    load() 
        read image from local image_path into self._image
    
    unload()
        clear self._image 
        
    save()
        write image stored in self._image to local image_path
    
        
    '''
    
    def __init__(self,
                 features=None,
                 instrument=None, 
                 framenumber=None,
                 image=None,
                 path=None,
                 info=None):
        self._instrument = instrument
        self._image = image
        self.framenumber = framenumber
        self.path = None
        self.image_path = None
        self.setDefaultPath(path)
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

    def setDefaultPath(self, path=None, imdir='norm_images/'):
        if path is None:
            path = self.image_path or self.path
        if not isinstance(path, str):               
            return
        
        if len(path) >= 4 and path[-4:] == '.png':  #### If path is a file, then set image_path 
            self.image_path = path        
            filename = path.split('/')[-1]
            if self.framenumber is None and len(filename) > 8:
                try:                        #### Try to read framenumber from end of path (i.e. 0107 from '...image0107.png')
                    self.framenumber = int(filename[-8:-4]) 
                except ValueError:
                    print('Warning - could not read integer framenumber from pathname')
            path = path.replace(filename, '')         #### remove filename (and norm_images/) to get base directory
            path = path.replace(imdir, '')
            self.path = path
        else:
            if '.' in path:    
                print('warning - {} is an invalid directory name'.format(path))
                return
            self.path = path               #### If path is a directory, set path and use framenumber to search for image_path                       
            if self.framenumber is not None:
                if path[-1] !='/':          
                    path += '/'
                filename = 'image' + str(self.framenumber).rjust(4, '0') + '.png'
                self.image_path = path + imdir + filename  
        if not os.path.isdir(self.path):  
                os.mkdir(self.path)      
            
    @property
    def image(self):
        return self._image if self._image is not None else cv2.imread(self.image_path)
    
    @image.setter    
    def image(self, image):
        if isinstance(image, np.ndarray):
            self._image = image 
    
    def load(self):
        self._image = self.image
                    
    def unload(self):
        self._image = None   
    
    def save(self):
        if self._image is not None and self.image_path is not None:
            filename = self.image_path.split('/')[-1]
            imdir = self.image_path.replace(filename, '')
            if not os.path.exists(imdir): 
                os.mkdir(imdir)
            cv2.imwrite(self.image_path, self._image)
        
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
            if isinstance(feature, Feature):   
                #### If we have a feature to add, set instrument and add with appropriate bbox
                if feature.model is not None and self.instrument is not None: 
                    feature.model.instrument = self.instrument
                if bbox not in self.bboxes:
                    self._features.append(feature)
                    self._bboxes.append(bbox)
            elif not feature:
                pass
            elif bbox is None:
                pass
            elif feature is not None:
                print('Warning - could not add feature {} of type {}'.format(i, type(feature)))
#                 msg = "features must be list of Features"
#                 msg += " or deserializable Features"
#                 raise(TypeError(msg))
    
    def remove(self, index):
        if isinstance(index, int):
            self._features.pop(index)
            bbox = self.bboxes.pop(index)
        elif isinstance(index, Feature):
            try:
                self.remove(self.features.index(index))
            except ValueError:
                print('Warning - could not remove Feature: Feature not found')
                return
        elif isinstance(index, list):
            for i in sorted(list(index), reverse=True): 
                self.remove(i)
        
    def optimize(self, report=True, **kwargs):
        for idx, feature in enumerate(self.features):
            result = feature.optimize(**kwargs)
            if report:
                print(result)

    def serialize(self, save=False, path=None, omit=[], omit_feat=[]):
        info = {}
        if 'features' not in omit:
            info['features'] = [feature.serialize( exclude=omit_feat ) for feature in self.features]
        if 'bboxes' not in omit:
            info['bboxes'] = self.bboxes
        if self.path is not None:
            info['path'] = self.path
        if self.image_path is not None:
            info['image_path'] = self.image_path
        if self.framenumber is not None:
            info['framenumber'] = str(self.framenumber)
        if save:
            if not os.path.exists(self.image_path): 
                self.save()

            framenumber = str(self.framenumber).rjust(4, '0') if self.framenumber is not None else ''
            path = path or self.path
            if path is None: 
                return info    
            if not (len(path) >= 5 and path[-5:] == '.json'):
                if not os.path.exists(path): os.mkdir(path)
                framenumber = str(self.framenumber).rjust(4, '0') if self.framenumber is not None else ''
                path += '/frame{}.json'.format(framenumber)
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
        if 'path' in info.keys():
            self.path = info(path)
        if 'image_path' in info.keys():                   
            self.image_path = info['image_path']                           
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
    
