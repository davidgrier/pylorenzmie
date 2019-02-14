"""
Author: Mark Hannel
"""

import numpy as np
import h5py

class TagArray(np.ndarray):

    def __new__(cls, input_array, frame_no=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        obj.frame_no = frame_no
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
        self.frame_no = getattr(obj, 'frame_no', None)

class h5video(object):
    def __init__(self, filename, mode='r'):
        self.filename = filename
        self.mode = mode
        self.image = None
        self.index = None
        self.dim = None
        self.nframes = None
        self.eof = False

    def __enter__(self):
        self.f = h5py.File(self.filename, self.mode)
        self.keys = self.f['images/'].keys()
        self.frames = self.f['images/'].values()
        self.nframes = len(self.frames)
        self.image = self.rewind()
        self.dim = self.f['images/' + self.keys[0]].shape
        return self

    def __exit__(self, *args):
        self.f.close()

    def get_image(self):
        try:
            self.image = self.frames[self.index]
            return self.image
        except IndexError:
            print("Index is out of range.")
            raise IndexError
            
    def get_time(self):
        return self.keys[self.index]

    def rewind(self):
        self.index = 0
        self.eof = False
        return self.get_image()

    def next(self):
        if self.eof:
            return None
        if self.index == self.nframes-2:
            self.eof = True
        self.index += 1
        return self.get_image()

    def goto(self, index):
        self.index = index

def example():
    import matplotlib.pyplot as plt

    # Grab h5 file.
    filename = 'example.h5'
    bg = np.load('example_bg.npy')

    with h5video(filename) as vid:
        plt.imshow(vid.frames[-30]/bg)
        plt.gray()
        plt.show()

        plt.imshow(vid.get_image()/bg)
        plt.gray()
        plt.show()

        vid.goto(220)
        
        plt.imshow(vid.get_image()/bg)
        plt.gray()
        plt.show()
        
        print('Example timestamp: {}'.format(vid.get_time()))
        print('Dimension of image: {}'.format(vid.dim))
        print('Number of frames: {}'.format(vid.nframes))
    
if __name__ == '__main__':
    example()
