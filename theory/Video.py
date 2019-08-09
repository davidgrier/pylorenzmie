# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import json
import trackpy as tp

##from CNNLorenzMie.Localizer import Localizer
##from CNNLorenzMie.Estimator import Estimator
##from CNNLorenzMie.nodoubles import nodoubles
from pylorenzmie.theory.Instrument import Instrument, coordinates
from pylorenzmie.theory.Frame import Frame
from pylorenzmie.theory.LMHologram import LMHologram as Model

import matplotlib.pyplot as plt


class Video(object):
    '''
Video Class: Abstraction of a Video (set of data taken with the same instrument).
               Handles prediction + estimation of initial particle parameters
               from images, and computations involving time evolution (i.e. trajectory)
               
    Attributes:
            detector (i.e. Localizer) : Detector() with method predict(images)
            which returns crop bounding box (i.e. x, y, w, h) for each feature
            from list of images

            estimator : Estimator() with method predict(images) which returns
            the z, a, n values from a list of images

            frames : List of Frame() objects, for storing + fitting Feature info


    Methods:
            add_frames() : predict initial feature parameters and add frames to Video

            to_df() : Return a DataFrame representation of the Video info

            trajectory() : Return a trajectory from Video info

            TODO : optimize()
    '''
    
    def __init__(self, images=[], detector=None, estimator=None, **kwargs):
##        self._detector = None  # should be YOLO
##        self.estimator = Estimator(model_path='CNNLorenzMie/keras_models/predict_stamp_fullrange_adamnew_extnoise_lowscale.h5') if estimator is None else estimator
##        self.instrument = self.estimator.instrument
        self.detector = detector
        self.estimator = estimator
        self.instrument = Instrument(**kwargs) if estimator is None else estimator.instrument
        self._frames = []
        self.add_frames(images)   
##        print(self.add_frames(images))

        
    @property
    def frames(self):
        return self._frames

    #### Predict+estimate initial parameters for images, and add corresponding frames to Video
    #### Input : list of images       Output : DataFrame with initial parameters
    def add_frames(self, images):
        SIZE = np.shape(images)[0]                                                 
        if(SIZE == 0):
            print('Warning: Video.add_frames was passed an empty list')
            return

        initial = pd.DataFrame()  ## Dataframe to keep track of output        

        # # First: get xy predictions, make frames, and return crops
        pred_list = self.detector.predict(images)  
        crops = []
        for i in range(SIZE):
            xy = pd.DataFrame(pred_list[i])['bbox']
            xy = pd.DataFrame(xy.tolist())
            xy.columns=['x_p', 'y_p', 'w', 'h']
            frm = Frame(images[i], xy, instrument=self.instrument)
            for j in xy.index:
                crops.append(frm.crop(j))
            self._frames.append(frm)
            xy['frame'] = i          ## Add xy predictions output
            initial = initial.append(xy, ignore_index=True)

        # # Next, use crops to estimate z, a, n, and add to dataframe and frames
        info = self.estimator.predict(img_list=crops)
        initial = initial.join(pd.DataFrame.from_dict(info))
        for i in range(SIZE):
            dfi = initial[initial.frame==i]  ## Slice of data at frame i
            dfi.index -= min(dfi.index)      ## Both indices have same ordering, but frame indexing starts at 0
            self._frames[i].df = self._frames[i].df.fillna(dfi)  ## Update vals

        return initial  ## Return the data sent to the frames

    #### Returns a DataFrame representation of the Video 
    def to_df(self):
        df = pd.DataFrame()
        SIZE = np.size(self.frames)
        for i in range(SIZE):
            dfi = self.frames[i].df
            dfi['frame'] = i
            df = df.append(dfi)
        return df

######    def read_df(self, df, images=None):
######        if images is None:
######            for i in range(np.size(self._frames)):
######                self._frames[i].df = df[df.frame==i]
######        else:
######            self._frames = []
######            for i in range(np.shape(images)[0]):
######                self._frames.append(Frame(images[i],

    #### Returns particle trajectories from data
    def trajectory(self):
        df = self.to_df().rename(columns={'x_p' : 'x', 'y_p' : 'y'})
        t = tp.link_df(df, 50, memory=3)
        return t.rename(columns={'x' : 'x_p', 'y' : 'y_p'})
                        
if __name__ == '__main__':
    '''
        For debugging, I wrote a substitute Estimator and Detector class, since
        CNN won't load on my local pc. The detector uses circletransform, and
        the estimator doesn't actually do anything; both output predictions in the
        same format as CNN Localizer and Estimator, in an effort to make it easier to
        eventually implement CNN. 
    '''

        
    class MyEstimator(object):   #### Object with predict() of same input/output
        def __init__(self, instrument):             ##  format as CNN-Estimator
            self.instrument = instrument
            
        def predict(self, img_list=[]): ## Doesn't actually predict; just gives predetermined                      
            z = 150*np.ones(np.shape(img_list)[0])## output of the same format, for testing
            a = 0.75/150*z
            n = 1.44/0.75*a
            z = z + 0.01*np.arange(np.shape(img_list)[0])
            return dict({'z_p': z.tolist(), 'a_p': a.tolist(), 'n_p': n.tolist()})
                
    class MyDetector(object): #### Detector. Uses circletransform and tp.batch,
        def detect(self, images):   ## and has same output format as YOLO localizer
            circles = []
            for n in range(np.shape(images)[0]):
                norm = images[n]
                circ = ct.circletransform(norm, theory='orientTrans')     
                circles.append(circ / np.amax(circ))
            return tp.batch(circles, 51, minmass=50) 

        def predict(self, images):  ## Returns detect() with YOLO formatting
            df = self.detect(images)[['x', 'y', 'frame']]           
            out=[]
            for i in range(np.shape(images)[0]):
                l = []
                for j in df[df.frame==i].index:
                    l.append(dict({'conf': '50%', 'bbox': (df.x[j], df.y[j], 201, 201)}))
                out.append(l)
            return out

    from pylorenzmie.theory.Feature import Feature
    import pylorenzmie.detection.circletransform as ct
    import cv2
    
    dark_count = 13
    PATH = 'pylorenzmie/tutorials/video_example/8hz_5V_t_const'

    background = cv2.imread('pylorenzmie/tutorials/video_example/background.png')
    background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    images = []
    cap = cv2.VideoCapture(PATH + '.avi')
    ret, image = cap.read()
    counter=0
    while(ret==True and counter<3):   ## Let's just take three frames to start
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        norm = (image - dark_count)/(background - dark_count)  
        images.append(norm)
        ret, image = cap.read()
        counter = counter+1

    det = MyDetector()                                  
    est = MyEstimator(Instrument())
    
    #### Now that we have images and a detector, let's make a Video object:
    print('lets make a video!')
    myvid = Video(images=images, detector=det, estimator=est)
    print('Video instance created!')
    print(myvid.to_df())
    print(myvid.trajectory())

    feat = myvid.frames[0].get_feature(0)
    myvid.frames[0].optimize(0)
    info = feat.serialize(filename='deep.json')
    print('serialized feat as:')
    print(feat.serialize(exclude=['data', 'coordinates', 'noise']))

    feat2 = Feature()
    feat2.model.instrument = feat.model.instrument
    feat2.deserialize('temp.json')
    print('loaded 2:')
    info2 = feat2.serialize(exclude=['data', 'coordinates', 'noise'])
    print(info2)
    print(pd.DataFrame(info2, index=[0]))
    ##feat2.read_df(pd.read_json('temp.json'))        

