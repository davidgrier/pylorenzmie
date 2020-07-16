#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2, json
import numpy as np
from .Feature import Feature
from ..Theory import coordinates

class Frame(object):
    '''
    Abstraction of an experimental video frame
    ...
    Attributes
    ----------
    image : numpy.ndarray
        Image from camera. Setter ensures the image is grayscale. Directly choosing an image sets image_path to None. 
    framenumber : int
    image_path : string    
        The setter will attempt to read the image from file at image_path. Additionally, if the Frame 
        does not have a framenumber, the setter will attempt to read one from the end of the string.
    bboxes : List of tuples ( {x, y, w, h} )
        Bounding box of dimensions (w, h) around feature at (x, y). Used for cropping to obtain image stamps
    features : List of Feature objects
        
    Methods
    ------- 
    add(features, info=None) : None
        features: (list of) Feature objects / list of serialized Features (dicts) / or list of bboxes (tuples)
        optional: info: (list of) serialized feature info (dicts)
            - Unpack and add (each) Feature to the Frame, 
            - if (each) input is a bbox, add it to Frame; otherwise, add None to frame's bbox list
            - deserialize info into Feature(s), if info passed
            
    crop(all) : None
        - Use each (not-None) bbox to crop the frame's image and update the corresponding feature's data
        all : boolean
            if all is False (default), then only update Features which do not already have data. 
        
    '''
    
    def __init__(self, features=None, instrument=None, 
                 framenumber=None, image=None, image_path=None, info=None):
        self._instrument = instrument
        self._framenumber = framenumber
        self._bboxes = []
        self._features = []
        self.image_path = image_path
        if self.image is None:
            self.image = image
        self.add(features)
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
        if not isinstance(path, String):
            self.image = None                                                   #### If invalid format, clear image and path
            if path is not None:
                print('Warning: could not read path of type {}'.print(type(path)))
            return
        if path[-1] is '/' and self.framenumber is not None:                    #### If path is a directory, look for image of format 'path/image(framenumber).png'
            path = path + 'image'+ str(self.framenumber).rjust(4, '0')+'.png'    
        self.image = cv2.imread(path, 0)                                        #### Read image (0 is grayscale)
        if self.image is not None:
            self._image_path = path                                             #### If an image was found, then keep the path
            if self.framenumber is None:                                        #### If path leads to an image and framenumber isn't set, try to
                self.framenumber = int(path[:-8:-4])                            ####    read framenumber from the path  (i.e. 0107 from '...image0107.png')                 
        else:
            print("Warning: image not found at path '{}'".format(path))

    @property
    def image(self):
        return self._image
    
    @image.setter(self):
    def image(self, image):
        self._image_path = None
        if not isinstance(image, np.ndarray):
            self._image = None
            if image is not None:
                print("Warning: could not read image of type {}".format(type(im)))
        elif len(np.shape(image)) is 2:
            self._image = image
        elif len(np.shape(image)) is 3:
            self._image = image[:, :, 0]
        else:
            self._image = None
            print("Warning: invalid image dimensions: {}".format(np.shape(image)))            
         self.crop(all=True) 
    
    @property
    def features(self):
        return self._features
    
    @property
    def bboxes(self):
        return self._bboxes
    
    def add(self, features, info=None): 
        if features is None:
            return
        features = [features] if not isinstance(features, list) else features                         #### Ensure input is a list
        info = [info for feature in features] if not isinstance(info, list) else info
        for i, feature in enumerate(features):
            bbox = None
            if isinstance(feature, tuple) and len(feature) is 4:    #### case where bbox passed
                bbox = feature
                feature = Feature()  
            elif isinstance(feature, dict):                         #### case where serialized feature passed
                feature = Feature().deserialize(info=feature)      
            if isinstance(feature, Feature):                        #### If a feature was found, add it
                self._bboxes.append(bbox)
                if self.instrument is not None:
                    feature.model.instrument = self.instrument
                if info[i] is not None:
                    feature.deserialize(info=info[i])
                self._features.append(feature)
            elif feature is not None:
                print('Warning: could not add feature {} of type {}'.format(i, type(feature)))
#                 msg = "features must be list of Features"
#                 msg += " or deserializable Features"
#                 raise(TypeError(msg))
        self.crop()                                                 #### find bboxes from features
    
    def crop(self, all=False):
        for i, bbox in enumerate(self.bboxes):
            if bbox is not None:
                feature = self._features[i]
                if all or feature.data is None:
                    if self.image is None:
                        feature.data = None
                    else:
                        (x, y, w, h) = bbox
                        center = ( int(np.round(x)), int(np.round(y)) )
                        crop_shape = (w, h)
                        cropped, corner = crop_center(self.image, center, crop_shape)
                        feature.data = cropped
                        feature.model.coordinates = coordinates(shape=crop_shape, corner=corner)
                        if feature.model.particle.x_p is None:
                            feature.model.particle.x_p = x
                        if feature.model.particle.y_p is None:
                            feature.model.particle.y_p = y
       
    def optimize(self, report=True, **kwargs):
        for idx, feature in enumerate(self.features):
            result = feature.optimize(**kwargs)
            if report:
                print(result)

    def serialize(self, filename=None, omit=[], omit_feat=[]):
        info = {}
        if 'features' not in omit:
            features = []
            bbox_info = []
            for i, feature in enumerate(self.features):
                if self.bboxes[i] is None:
                    features.append(feature.serialize( exclude=omit_feat ))
                else:
                    bbox_info.append(feature.serialize( exclude=['data'].extend(omit_feat) )) ))
            info['features'] = features
            info['bbox_info'] = [(x if len(x.keys()) > 0 else None) for x in bbox_info] 
        
        if 'bboxes' not in omit:
            bboxes = []
            for bbox in self.bboxes:
                if bbox is not None:
                    bboxes.append(bbox)
            info['bboxes'] = bboxes
        if self.image_path is not None:
            info['image_path'] = self.image_path
        if self.framenumber is not None:
            info['framenumber'] = str(self.framenumber)
#         for k in omit:
#             if k in info.keys():
#                 info.pop(k)
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
        if 'framenumber' in info.keys():
            self.framenumber = int(info['framenumber']) 
        else:
            self.framenumber = None
        if 'image_path' in info.keys():
            self.image_path = image_path
        else:
            self.image = None
        if 'features' in info.keys():          #### Add any features passed in serial form
            self.add(info['features'])
        if 'bboxes' in info.keys():            #### Add any features specified by bboxes
            bboxes = info['bboxes']
            if 'bbox_info' in info.keys():
                self.add_bbox(bboxes, info=info['bbox_info'])
            else:
                self.add_bbox(bboxes)


                
#### Static helper method. Literally copy-pasted from Lauren Altman's crop_feature - can probably just import it instead in the future                                 
def crop_center(img_local, center, cropshape):
    (xc, yc) = center
    (crop_img_rows, crop_img_cols) = cropshape
    (img_cols, img_rows) = img_local.shape[:2]
    if crop_img_rows % 2 == 0:
        right_frame = left_frame = int(crop_img_rows/2)
    else:
        left_frame = int(np.floor(crop_img_rows/2.))
        right_frame = int(np.ceil(crop_img_rows/2.))
    xbot = xc - left_frame
    xtop = xc + right_frame
    if crop_img_cols % 2 == 0:
        top_frame = bot_frame = int(crop_img_cols/2.)
    else:
        top_frame = int(np.ceil(crop_img_cols/2.))
        bot_frame = int(np.floor(crop_img_cols/2.))
    ybot = yc - bot_frame
    ytop = yc + top_frame
    if xbot < 0:
        xbot = 0
        xtop = crop_img_rows
    if ybot < 0:
        ybot = 0
        ytop = crop_img_cols
    if xtop > img_rows:
        xtop = img_rows
        xbot = img_rows - crop_img_rows
    if ytop > img_cols:
        ytop = img_cols
        ybot = img_cols - crop_img_cols
    cropped = img_local[ybot:ytop, xbot:xtop]
    corner = (xbot, ybot)
    return cropped, corner
                                  
                               
            
