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
        self.child = None
        self.shape = shape
        self.order = order
        self.index = 0
        self._initialized = False
        self._cycled = False

    def filter(self, data):
        self.add(data)
        return self.get() # if self.initialized else data

    def get(self, reshape=True):
        '''Return current median image

        :returns: median image
        :rtype: numpy.ndarray
        '''
        return self._data.reshape(self.shape) if reshape else self._data

    def add(self, data):
        '''include a new image in the median calculation

        :param data: image data
        :returns:
        :rtype:
        '''
        if data.shape != self.shape:
            self._data = data.astype(np.uint8).ravel()
            self.shape = data.shape
        if self.order == 0:
            self.buffer[self.index, :] = data.astype(np.uint8).ravel()
            self.index += 1
        else:
            child = self.child
            child.add(data)
            if child.initialized:
                self.buffer[self.index, :] = child.get(reshape=False)
            if child.cycled:
                self.index += 1
        if self.index == 3:
            self.index = 0
            self._data = np.median(self.buffer, axis=0).astype(np.uint8)
            self._initialized = True
            self._cycled = True
        else:
            self._cycled = False

    def reset(self):
        self._initialized = False
        self._cycled = False
        if self.order > 0:
            self.child.reset()

    @property
    def initialized(self):
        return self._initialized

    @property
    def cycled(self):
        return self._cycled

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, shape):
        self._shape = shape
        if shape is None:
            return
        if self.child is not None:
            self.child.shape = shape
        self.npts = np.product(shape)
        self.buffer = np.zeros((3, self.npts), dtype=np.uint8)
        self.index = 0
        self._initialized = False

    @property
    def order(self):
        return self._order

    @order.setter
    def order(self, order):
        self._order = np.clip(order, 0, 10)
        if (self._order > 0):
            self.child = vmedian(order=self._order-1, shape=self.shape)
        self._initialized = False
