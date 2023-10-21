import h5py
import numpy as np

import logging
logging.basicConfig()
logger = logging.getLogger('configuration')
logger.setLevel(logging.INFO)


class h5video(object):
    '''Class for reading HDF5 videos created by pyfab'''
    def __init__(self, filename):
        self.filename = filename
        self.image = None
        self.index = None
        self.shape = None
        self.nframes = 0
        self.eof = False

    def __enter__(self):
        self.h5file = h5py.File(self.filename, 'r')
        self.keys = self.h5file['images/'].keys()
        self.frames = self.h5file['images/'].values()
        self.nframes = len(self.frames)
        self.image = self.rewind()
        self.shape = self.h5file['images/' + self.keys[0]].shape
        return self

    def __exit__(self, *args):
        self.h5file.close()

    def get_image(self):
        try:
            self.image = self.frames[self.index]
            return self.image
        except IndexError:
            msg = 'Index {} is out of range ({})'
            logger.warn(msg.format(self.index, self.nframes))
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

        print(f'Example timestamp: {vid.get_time()}')
        print(f'Dimension of image: {vid.shape}')
        print('Number of frames: {vid.nframes}')


if __name__ == '__main__':
    example()
