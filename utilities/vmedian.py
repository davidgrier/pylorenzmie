import numpy as np


class vmedian(object):

    def __init__(self, order=0, dimensions=None):
        """Compute running median of a video stream

        :param order: depth of median filter: 3^(order + 1) images
        :param dimensions: (width, height) of images
        :returns: 
        :rtype: 

        """
        self.child = None
        self.dimensions = dimensions
        self.order = order
        self.initialized = False
        self.index = 0

    def filter(self, data):
        self.add(data)
        return self.get()
    
    def get(self):
        """Return current median image

        :returns: median image
        :rtype: numpy.ndarray

        """
        return np.median(self.buffer, axis=0)
    
    def add(self, data):
        """include a new image in the median calculation

        :param data: image data
        :returns: 
        :rtype: 

        """
        if isinstance(self.child, vmedian):
            self.child.add(data)
            if (self.child.index == 0):
                self.buffer[self.index, :, :] = self.child.get()
                self.index = self.index + 1
        else:
            self.buffer[self.index, :, :] = data
            self.index = self.index + 1
            
        if self.index == 3:
            self.index = 0
            self.initialized = True
            
    @property
    def dimensions(self):
        return self._dimensions

    @dimensions.setter
    def dimensions(self, dimensions):
        if dimensions is not None:
            self.buffer = np.zeros((3, dimensions[0], dimensions[1]),
                                   dtype=np.uint8)
            self.index = 0
            self._dimensions = dimensions
            self.initialized = False
            if isinstance(self.child, vmedian):
                self.child.dimensions = dimensions
            
    @property
    def order(self):
        return self._order

    @order.setter
    def order(self, order):
        self._order = np.clip(order, 0, 10)
        if (self._order == 0):
            self.child = None
        else:
            if isinstance(self.child, vmedian):
                self.child.order = self._order - 1
            else:
                self.child = vmedian(order=self._order - 1,
                                     dimensions=self.dimensions)
        self.initialized = False
