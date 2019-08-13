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
            set_frames() : predict initial feature parameters and add frames to Video

            seialize() : Return a DataFrame representation of the Video object.
                    If filename is given, save Video to .json file as dict

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
        self.set_frames(images)   
##        print(self.add_frames(images))

        
    @property
    def frames(self):
        return self._frames

    #### Predict+estimate initial parameters for images, and add corresponding frames to Video
    #### Input : list of images       Output : DataFrame with initial parameters
    def set_frames(self, images):
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
            frm = Frame(xy, image=images[i], instrument=self.instrument)
            for j in xy.index:
                crops.append(frm.crop(j))
            self._frames.append(frm)
            xy['frame'] = i          ## label xy predictions by frame for output
            initial = initial.append(xy, ignore_index=True)

        # # Next, use crops to estimate z, a, n, and add to dataframe and frames
        info = self.estimator.predict(img_list=crops)
        initial = initial.join(pd.DataFrame.from_dict(info))
        for i in range(SIZE):
            dfi = initial[initial.frame==i]  ## Slice of data at frame i
            dfi.index -= min(dfi.index)      ## Both indices have same ordering, but frame indexing starts at 0
            self._frames[i].df = self._frames[i].df.fillna(dfi)  ## Update vals

        return initial  ## Return the data sent to the frames

    #### Returns a DataFrame representation of the Video.
    #### Save the video as a json dict at filename, if included
    def serialize(self, filename=None):
        df = pd.DataFrame()
        SIZE = np.size(self.frames)
        for i in range(SIZE):
            self.frames[i].df['frame'] = i
            dfi = self.frames[i].df
            df = df.append(self.frames[i].df, ignore_index=True)
            self.frames[i].df = self.frames[i].df.drop(columns=['frame'])
            
        if filename is not None:
            info = df.to_dict()
            shape = []
            data = []
            for i in range(SIZE):
                im = self.frames[i].image
                shape.append(np.shape(im))
                data.append(im.reshape(np.size(im)).tolist())
            info['shape'] = shape
            info['data'] = data
            with open(filename, 'w') as f:
                json.dump(info, f)
            print('Wrote Video as ' + filename )
        return df
        
    #### Restore Video object from a json dict
    #### Alternatively restore dict from named file.
    #### To restore images to frames, either pass a list of images, or
    #### include the data in the dict with key 'data'
    def deserialize(self, info, images=None):
        if info is None:
            return  
        if isinstance(info, str):
            with open(info, 'rb') as f:
                info = json.load(f)

        shape = []
        if images is not None:
            for i in np.shape(images)[0]:
                shape.append(np.shape(images[i]))
        else:
            if 'data' in info.keys():
                images = info['data']
                shape = info['shape']
                del info['data']
                del info['shape']
            else:
                print("Warning: No image data provided; Frames created from deserialize without images")
                
        self._frames = []
        df = pd.DataFrame.from_dict(info)
        for i in range(np.max(df.frame) + 1):
            dfi = df[df.frame==i].drop(columns=['frame'])
            dfi.index = range(np.size(dfi.index))
            self._frames.append(Frame(dfi,
                                      image=np.array(images[i]).reshape(shape[i]),
                                      instrument=self.instrument))
##            self._frames.append(Frame(df[df.frame==i].drop(columns=['frame']),
##                                      image=np.array(images[i]).reshape(shape[i]),
##                                      instrument=self.instrument))


    #### framelist is a list of frames to optimize, featlist is a list of lists of features to optimize
    ####i.e. optimize([0, 2], [ [0, 2], [1] ]) will optimize features 0 and 2 in frame 0 and feature 1 in frame 1        
    def optimize(self, framelist=[], featlist=[]):
        for i in range(np.size(framelist)):
            print( "optimizing in frame " + str(i))
            self._frames[framelist[i]].optimize(featlist[i])
        
    #### Returns particle trajectories from data
    def trajectory(self):
        df = self.serialize().rename(columns={'x_p' : 'x', 'y_p' : 'y'})
        t = tp.link_df(df, 50, memory=3)
        return t.rename(columns={'x' : 'x_p', 'y' : 'y_p'})
                        
if __name__ == '__main__':
    '''
        For debugging, I wrote a substitute Estimator and Detector class, since
        CNN won't load on my local pc. The detector uses circletransform, and
        the estimator doesn't actually do anything; both output predictions in the
        same format as CNN Localizer and Estimator, in an effort to make it easier to
        eventually implement CNN.

        To run main, move Video.py to the same directory as pylorenzmie 
    '''

        
    class MyEstimator(object):   #### Object with predict() of same input/output
        def __init__(self, instrument):             ##  format as CNN-Estimator
            self.instrument = instrument
            self.instrument.dark_count = 13
            self.instrument.background = 1.
            
        def predict(self, img_list=[]): ## Doesn't actually predict; just gives predetermined                      
            z = 150*np.ones(np.shape(img_list)[0])## output of the same format, for testing
            a = 1.5/150*z
            n = 1.44/1.5*a
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
    import matplotlib.pyplot as plt


#### First, let's make a detector and an estimator
    det = MyDetector()                                  
    est = MyEstimator(instrument=Instrument(wavelength=.447,    #pixels
                                   magnification=.048, #microns/pixel
                                   n_m=1.340,
                                   dark_count=13,
                                   background=1.))


    dark_count = 13
    PATH = 'pylorenzmie/tutorials/video_example/8hz_5V_t_const'


######  Next, let's get the images from the video and make a Video object:
##    
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


    print('lets make a video!')
    myvid = Video(images=images, detector=det, estimator=est)
    print('Video instance created!')
    
##    print('lets load a video . . .')
##    myvid = Video(detector=det, estimator=est)
##    myvid.deserialize('vid.json')
    

    print(' - - - ')
##    print('Old Optimization Method')
##
##    xc = 100
##    yc = 100
##    crop = myvid._frames[1].crop(0)
##    plt.imshow(crop, cmap='gray')
##    
##    guesses = {'a_p': 0.75,      'n_p': 1.44,     'r_p': [100., 100., 150.]}
##    myFt = Feature(**guesses)
##    myFt.model.instrument = Instrument(wavelength=.447,    #pixels
##                                   magnification=.048, #microns/pixel
##                                   n_m=1.340,
##                                   dark_count=13,
##                                   background=1.)
##
##    myFt.model.coordinates=coordinates(np.shape(crop))
##    myFt.data=crop.reshape(np.size(crop))
##    print('coordinates')
##    print(myFt.coordinates)
##    print(myFt.model.coordinates)
##    print('old optimize: ')
##    print('pre data')
##    print(myFt.serialize(exclude=['data', 'coordinates', 'noise']))
##    myFt.optimize()
##    print('post data')
##    print(myFt.serialize(exclude=['data', 'coordinates', 'noise']))
##    print('frame object optimization')


## Immediately we can do things like compute the trajectory of the particle
##  from the initial guesses:
    print(myvid.trajectory())

#### Usually, the guesses are accurate enough to link the correct particles
#### together, no fitting necessary.
    
#### We can also run optimizations on an individual frame. We can run fits on
#### selected features:
    frame = myvid.frames[1]
    print('Frame before fitting:')
    print(frame.df)
    frame.optimize([0, 1]) ## Optimize features 0 and 1
## Note: Frame.optimize() is compatible with many different types of syntax:
##    frame.optimize() ## Optimize all features in Frame
##    frame.optimize(0) ## optimize frame 0 only
##    frame.optimize([1]) ## optimize frame 1 only
    print('Frame after fitting:')
    print(frame.df)

#### Or, we can run optimizations on select features from select frames:
    print('Video before fitting:')
    print(myvid.serialize())
    myvid.optimize(framelist=[0, 2], featlist=[ [1], [0, 1] ])
    print('Video after fitting:')
    print(myvid.serialize())

#### We can also save a Video to a .json file ...
    myvid.serialize(filename=PATH+'_myvid.json')
#### ... and restore it into an empty Video later
    newdet = MyDetector()                                  
    newest = MyEstimator(instrument=Instrument(wavelength=.447,    #pixels
                                   magnification=.048, #microns/pixel
                                   n_m=1.340,
                                   dark_count=13,
                                   background=1.))
    newvid = Video(detector=newdet, estimator=newest)
    newvid.deserialize(PATH+'_myvid.json')
    print(newvid.serialize())
    
