#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2, json
import numpy as np
import os
import pandas as pd
from .Feature import Feature

class Frame(object):
    '''
    Abstraction of an experimental video frame
    ...
    Attributes
    ----------
    image_path : string    
    image : numpy.ndarray
        Image from camera. If None (default), the getter tries to read from local image_path. To save image, call load()
    framenumber : int
    features : List of Feature objects
    bboxes : List of tuples ( {x, y, w, h} )
        Bounding box of dimensions (w, h) around feature at (x, y). Used for cropping to obtain image stamps
    
        
    Methods
    ------- 
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
                 framenumber=None, image=None, image_path=None, info=None):
        self._instrument = instrument
        self._framenumber = framenumber
        self._image = None
        self.image = image
        self.image_path = image_path
        if self.image_path is None: print('Warning - image path not set')
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
    
    @property
    def framenumber(self):
        return self._framenumber

    @framenumber.setter
    def framenumber(self, idx):
        self._framenumber = idx
    
    @property
    def image_path(self):
        return self._image_path
    
    @image_path.setter                   
    def image_path(self, path):
        if not isinstance(path, str):                                           #### Catch invalid format
            self._image_path = None
            if path is not None:
                print('Warning - could not read path of type {}'.print(type(path)))
            return
        if path[-1] is '/' and self.framenumber is not None:                    #### If path is a directory, look for image of format 'path/image(framenumber).png'
            path = path + 'image'+ str(self.framenumber).rjust(4, '0')+'.png'    
        if os.path.exists(path):
            self._image_path = path                                             #### If an image was found, then keep the path
            print('file found - set path to '+path)
            if self.framenumber is None:                                        #### If path leads to an image and framenumber isn't set, try to
                try:
                    self.framenumber = int(path[-8:-4])                             ####    read framenumber from the path  (i.e. 0107 from '...image0107.png')
                    print('read framenumber {} from path'.format(self.framenumber))
                except ValueError:
                    print('Warning - could not read integer framenumber from pathname')
        else:
            self._image_path = None
            print("Warning - invalid path: "+path)
           
    @property
    def image(self):
        return self._image if self._image is not None else cv2.imread(self.image_path)
                
    def load(self):
        self._image = self.image
        if isinstance(self._image, np.ndarray):
            print('Successfully loaded image from file path')
        elif self._image is None:
            print('Warning - failed to load image from path '+self.image_path)
        else:
            print('Warning - Invalid image format: setting to None')
            self.unload()
                    
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
    
    def remove(self, indices):
        for i in sorted(list(indices), reverse=True):
            print(i)
            self._features.remove(i)
            self._bboxes.remove(i)
        
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
        if self.image_path is not None:
            info['image_path'] = self.image_path
        if self.framenumber is not None:
            info['framenumber'] = str(self.framenumber)
        save = save if path is None else True
        if save and self.image_path is not None:
            path = self.image_path.split('_norm_images/')[0] if path is None else path
            path += '_frames/'
            if not os.path.exists(path):
                os.makedirs(path)
            filename = path + 'frame' + str(self.framenumber).rjust(4, '0') + '.json'
            with open(filename, 'w') as f:
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
            self.image_path = info['image_path']                           #### Note: setting image path will call image setter
        features = info['features'] if 'features' in info.keys() else []
        bboxes = info['bboxes'] if 'bboxes' in info.keys() else []
        self.add(features=features, bboxes=bboxes) 
    
    def to_df(self, info=None):
        info = info or self.serialize(omit_feat=['data'])
        df = pd.DataFrame()
        features = info['features'] if 'features' in info.keys() else {}
        bboxes = info['bboxes'] if 'bboxes' in info.keys() else []
        df = pd.DataFrame.from_dict(features)
        df['bboxes'] = bboxes
        df['framenumber'] = info['framenumber'] if 'framenumber' in info.keys() else None  
        return df

    
    


# if __name__ == '__main__':    


#     img_path = 'examples/test_image_crop_201.png'
#     img = cv2.imread(img_path)
#     estimator = Estimator(model_path=keras_model_path, config_file=config_json)
#     data = estimator.predict(img_list = [img], scale_list=[1])
#     example_z = round(data['z_p'][0],1)
#     example_a = round(data['a_p'][0],3)
#     example_n = round(data['n_p'][0],3)
#     print('Example Image:')
#     print('Particle of size {}um with refractive index {} at height {}'.format(example_a, example_n, example_z))
    

