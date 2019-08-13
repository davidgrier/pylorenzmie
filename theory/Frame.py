# -*- coding: utf-8 -*-

import timeit
import numpy as np
import pandas as pd
from pylorenzmie.theory.Instrument import Instrument, coordinates
from pylorenzmie.theory.Feature import Feature
from pylorenzmie.theory.LMHologram import LMHologram as Model
import matplotlib.pyplot as plt



####
class Frame(object):
    '''
Frame Class: Abstraction of a video frame (image of data). Handles image cropping
             and feature management/fitting.
             
Attributes:
    df : dataframe containing information about features. Each row corresponds to
           a different feature. 
           columns: [ x_p, y_p, z_p, a_p, n_p, k_p,      w, h,         optimized  ]
                     |     particle.properties     | crop dimensions |  boolean  |             

    image: image corresponding to frame

    instrument : Instrument() object

Methods:
        crop(i) : Return the cropped image corresponding to i'th feature.
                NOTE: the crop bounding box is taken from df as x, y, w, h

        get_feature(i) : Returns a complete Feature object of the i'th  
        
        optimize(index) : Create features, run fitting, and update df accordingly

   '''

    ## Initialize a new Frame.
    ##      Required: a dataframe initial, with # rows = # features.
    ##          info in initial will be added to df, but initial can be empty.
    def __init__(self, initial, image=None, instrument=None, **kwargs):
        self.df = pd.DataFrame(columns=Feature().model.particle.properties, index=initial.index)
        self.df = self.df.join(pd.DataFrame(columns=['w', 'h']))
        self.df['optimized'] = False
        self.df = self.df.fillna(initial)
        self.image = image
        self.instrument = Instrument(**kwargs) if instrument is None else instrument

    ## Return crop(s) of image for features in index from x_p, y_p, w, h in df
    ## index=int -> return crop of index'th feature
    ## index=list of n int -> return list of n crops
    ## index=None -> return list of all crops
    def crop(self, index=None):
        if index is None:
            INDEX = self.df.index
        elif isinstance(index, int):
            INDEX = [index]
        else:
            INDEX = index

        if self.image is None:
            print("error in Frame.crop(): Frame has no image")
            return None
        
        crops = []
        for i in INDEX:
            x = int(self.df.x_p[i])
            y = int(self.df.y_p[i])
            w = int(self.df.w[i])
            h = int(self.df.h[i])
            crops.append(self.image[y-h//2:y+h//2, x-w//2:x+w//2])
        if isinstance(index, int):
            crops = crops[0]
        return crops
    
    ##Return i'th feature. TODO: return list of features.
    def get_feature(self, i):  
        f = Feature()
        f.deserialize(self.df.iloc[i].to_dict())
        f.data = self.crop(i).reshape(np.size(self.crop(i)))
        return f


    ## Optimize index'th feature(s) and update df accordingly.
    ##      Because all of the features share the same type of model, and
    ##      creating a new Feature (and hence a new Model) is slow, we use only
    ##      one Feature instance for efficiency.
    ##      If optimize() is called from a Video object, the process can be made
    ##      even more efficient by passing the same Feature in the Video to
    ##      multiple Frames
    ##
    ## index=int -> fit one feature
    ## index=list of n int -> fit n features
    ## index=None -> fit all features
    def optimize(self, index=None, feature=None):
        if index is None:
            index = self.df.index
        elif isinstance(index, int):
            index = [index]
        
        if feature is not None:  ## If a feature was passed, then use it
            f = feature
        else:
            f = Feature()               
            f.model.instrument = self.instrument
##        f.mask.settings['percentpix'] = 1.  ## for debugging
        for i in index:
            print("optimizing feature " + str(i))
                                                      ## Send dataframe info 
            f.deserialize(self.df.loc[i].to_dict())  ## to feature object
##            f.model.instrument = self.instrument
            
            crop = self.crop(i)                       ## Get crop for fitting
            f.model.coordinates = coordinates(np.shape(crop))
            f.data = crop.reshape(np.size(crop))    
            x0 = self.df.x_p[i]
            y0 = self.df.y_p[i]
            xc = self.df.w[i] // 2
            yc = self.df.h[i] // 2
            f.model.particle.x_p = xc   ## We guess that the particle is 
            f.model.particle.y_p = yc   ## in the center of the crop

            
##            if(f.model.particle.k_p == nan):
##                f.particle.k_p = 0.0
            f.model.particle.k_p = 0.0                ## Default k_p = 0
##            print(' Pre-Optimized Feature:')
##            print(f.serialize(exclude=['data', 'coordinates', 'noise']))
            f.optimize()
##            print(' Post-Optimized Feature:')
##            print(f.serialize(exclude=['data', 'coordinates', 'noise']))
            info = f.serialize(exclude=['data', 'coordinates', 'noise'])
            df=pd.DataFrame(info, index=[i])
            df['optimized']=True
            for label in df.columns:
                if label in self.df.columns:
                    self.df.at[i, label] = df.at[i, label]
            self.df.at[i, 'x_p'] += x0 - xc
            self.df.at[i, 'y_p'] += y0 - yc
