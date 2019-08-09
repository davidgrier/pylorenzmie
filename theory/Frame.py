# -*- coding: utf-8 -*-

import timeit
import numpy as np
import pandas as pd
from pylorenzmie.theory.Instrument import Instrument, coordinates
from pylorenzmie.theory.Feature import Feature
from pylorenzmie.theory.LMHologram import LMHologram as Model

####
class Frame(object):
        '''
Frame Class: Abstraction of a video frame (image of data). Handles image cropping
                and handles feature management/fitting 
    Attributes:
            image: image corresponding to frame

            instrument : Instrument() object

            df : dataframe with feature parameters (r_p, a_p, n_p, etc) as well as
                  crop dimensions (w, h) and boolean 'optimize' (whether or not row)
                  was fitted yet). Each column corresponds to a feature

    Methods:
            crop(i) : Return the cropped image corresponding to i'th feature.
                    NOTE: the crop bounding box is taken from df as x, y, w, h

            get_feature(i) : returns a complete Feature object of the i'th  
            
            TODO : optimize(). (Should create features, fit them, and update df.) 
    '''
    
    def __init__(self, image, initial_xy, instrument=None, **kwargs):
        self.image = image
        self.df = initial_xy
##        self.df = self.df.join(pd.DataFrame(columns=['z_p', 'a_p', 'n_p']))
        self.df = self.df.join(pd.DataFrame(columns=['z_p', 'a_p', 'n_p', 'k_p']))
        self.instrument = Instrument(**kwargs) if instrument is None else instrument
        self.df['optimized'] = False

    def _crop(self, i):     ## Return crop of i'th feature (private)
        x = int(self.df.x_p[i])
        y = int(self.df.y_p[i])
        w = self.df.w[i]
        h = self.df.h[i]
        return self.image[y-h//2:y+h//2, x-w//2:x+w//2]

    def crop(self, index=None):  ## Return list containing crop(s) of features in index
        if isinstance(index, int):
            return self._crop(index)

        crops = []
        index = self.df.index if index is None else index
        for i in index:
            crops.append(self._crop(i))
        return crops
        
    def get_feature(self, i):  ##Return i'th feature. TODO: return list of features.
        f = Feature(instrument=self.instrument)
        f.deserialize(self.df.iloc[i].to_dict())
        f.data = self.crop(i).reshape(np.size(self.crop(i)))
        return f

####    def optimize(self, index=[]):
####        f = Feature()
####        f.model.instrument = self.instrument
####        
####        if isinstance(index, int):
####            index = [index]
####
####        for i in index:
####            f.deserialize(self.df.iloc[i].to_dict())
####            f.data = self.crop(i).reshape(np.size(self.crop(i)))
####            print(f.coordinates)
####            f.model.coordinates = f.coordinates
####            f.optimize()
####            info = f.serialize()
####            print(info)
####            df=pd.DataFrame.from_dict(info)
####            df['optimize']=True
####            print(df)
####            print(self.df)
####            self.df.iloc[i] = df
####            print(self.df)
           
