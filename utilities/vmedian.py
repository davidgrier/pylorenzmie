# -*- coding: utf-8 -*-

"""Efficient approximation to a running median filter."""

import numpy as np


class vmedian(object):

    def __init__(self, order=0, shape=None):
        """Compute running median of a video stream

        :param order: depth of median filter: 3^(order + 1) images
        :param dimensions: (width, height) of images
        :returns:
        :rtype:

        """
        self.children = []
        self.shape = shape
        self.order = order
        self.index = 0

    def filter(self, data):
        self.add(data)
        return self.get()

    def get(self, reshape=True):
        """Return current median image

        :returns: median image
        :rtype: numpy.ndarray

        """
        data = np.median(self.buffer, axis=0).astype(np.uint8)
        return np.reshape(data, self.shape) if reshape else data

    def add(self, data):
        '''include a new image in the median calculation

        :param data: image data
        :returns:
        :rtype:

        '''
        if data.shape != self.shape:
            self.shape = data.shape
        if self.order == 0:
            self.buffer[self.index, :] = np.ravel(data)
            self.index += 1
        else:
            child = self.children[self.index]
            child.add(data)
            if (child.index == 0):
                self.buffer[self.index, :] = child.get(reshape=False)
                self.index += 1
        if self.index == 3:
            self.index = 0
            self.initialized = True

    def reset(self):
        self.initialized = False
        if isinstance(self.child, vmedian):
            self.child.reset()

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, shape):
        self._shape = shape
        if shape is None:
            return
        self.npts = np.product(shape)
        self.buffer = np.zeros((3, self.npts), dtype=np.uint8)
        self.index = 0
        self.initialized = False
        for child in self.children:
            child.shape = shape

    @property
    def order(self):
        return self._order

    @order.setter
    def order(self, order):
        self._order = np.clip(order, 0, 3)
        if (self._order == 0):
            self.children = None
        else:
            self.children = [vmedian(order=self._order-1,
                                     shape=self.shape)
                             for _ in range(3)]
        self.initialized = False
