#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2, json
import numpy as np
from .Feature import Feature

class Frame(object):
"""
    Object representation of an experimental video frame. Frames can have an image (data), a framenumber, an instrument for fitting, a list of Feature objects
    and a corresponding list of bounding boxes. Features can be added in two ways: 
      1) If the frame has an image, then Features are added by specifying a bbox (via a dict) in deserialize . The bbox specifies x_p and y_p, and 
         Feature data (cropped image) is obtained by using the bbox to crop the Frame data. Optionally, the dict can also pass other feature info 
         (like z_p, a_p, etc.)  through a dict 'bbox_info'
      2) Feature objects can be passed directly. Their corresponding bbox will be 'none' and are serialized individually (under 'features') with their own data.
   
"""
    def __init__(self, features=None, instrument=None, 
                 framenumber=None, image=None, info=None):
        self._instrument = instrument
        self._framenumber = framenumber
        self.image = image
        self._bboxes = []
        self._features = []
        if features is not None:
            self.add(features)
        if info is not None:
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
    
    @image_path.setter                   #### Method for reading an image path into a frame
    def image_path(self, path):
        image = cv2.imread(path, 0)      #### 0 is grayscale
        if image is not None:            #### If this path is valid, then read framenumber for the path (unless Frame already has a framenumber) 
            self.framenumber = int(path[:-8:-4]) if self.framenumber is not None else self.framenumber
        elif self.framenumber is not None:
            image = cv2.imread(path + 'image'+ str(self.framenumber).rjust(4, '0')+'.png') 
        if image is not None
            self.image = image
            self._image_path = path
        else:
            self._image_path = None
            print("Warning: Could not read image from path '{}'".format(im))
            
    @property
    def image(self):
        return self._image
    
    @image.setter(self):
    def image(self, image):
        if isinstance(image, String)
            self.image_path = image    
        elif isinstance(image, np.ndarray):
            self._image_path = None
            if len(np.shape(image)) is 2:
                self._image = image
            elif len(np.shape(image)) is 3:
                self._image = image[:, :, 0]
            else:
                self._image = None
                print("Warning: invalid image dimensions: {}".format(np.shape(image)))            
        else:
            self._image = None
            self._image_path = None
            if image is not None:
                print("Warning: invalid image format: {}".format(type(im)))
                
    @property
    def features(self):
        return self._features
    
    @property
    def bboxes(self):
        return self._bboxes
    
    def add(self, features): 
        if type(features) is Feature:
            features = [features] 
        for feature in features:
            if self.instrument is not None:
                feature.model.instrument = self.instrument
            if type(feature) is Feature:
                self._features.append(feature)
                self._bboxes.append(None)
            else:
                msg = "features must be list of Features"
                msg += " or deserializable Features"
                raise(TypeError(msg))
                
    def add_bbox(self, bboxes, info=None):
        info = info or [None for bbox in bboxes]
        for i, bbox in enumerate(bboxes):
            feature = Feature(data=self.crop(bbox))
            feature.x_p = bbox[0]
            feature.y_p = bbox[1]
            feature.deserialize(info[i])
            self._features.append(feature)
            self._bboxes.append(bbox)         

    def crop(self, bbox):
        if self.image is None:
            return None
        (x, y, w, h) = bbox
        center = ( int(np.round(x)), int(np.round(y)) )
        cropshape = (w, h)
        cropped, corner = crop_center(self.image, center, cropshape)
        return cropped
    
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
                                  
                               
            
