'''Efficient approximation to a running median filter.'''

import numpy as np


class VMedian:
    '''Running median of a video stream.

    Computes an approximate running median using a hierarchical tree of
    3-element median buffers.  Each level reduces the update rate by a
    factor of 3, so a tree of depth ``order`` accumulates
    ``3**(order + 1)`` frames before producing its first output.

    Parameters
    ----------
    order : int
        Depth of the median tree. Default: 0 (single 3-frame buffer).
    shape : tuple of int, optional
        Shape ``(height, width)`` of the input images.
    '''

    def __init__(self, order: int = 0, shape: tuple | None = None) -> None:
        self.child = None
        self.shape = shape
        self.order = order
        self.index = 0
        self._initialized = False
        self._cycled = False

    def filter(self, data: np.ndarray) -> np.ndarray:
        '''Add *data* and return the current median estimate.'''
        self.add(data)
        return self.get()

    def get(self, reshape: bool = True) -> np.ndarray:
        '''Return the current median image.

        Parameters
        ----------
        reshape : bool
            If True (default), return array with the original image shape.
            If False, return the flattened internal buffer.

        Returns
        -------
        median : ndarray
        '''
        return self._data.reshape(self.shape) if reshape else self._data

    def add(self, data: np.ndarray) -> None:
        '''Include a new image in the median calculation.

        Parameters
        ----------
        data : ndarray
            New image frame.
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

    def reset(self) -> None:
        '''Reset the filter state.'''
        self._initialized = False
        self._cycled = False
        if self.order > 0:
            self.child.reset()

    @property
    def initialized(self) -> bool:
        '''True once enough frames have been accumulated.'''
        return self._initialized

    @property
    def cycled(self) -> bool:
        '''True if the buffer completed a cycle on the last :meth:`add`.'''
        return self._cycled

    @property
    def shape(self) -> tuple | None:
        return self._shape

    @shape.setter
    def shape(self, shape: tuple | None) -> None:
        self._shape = shape
        if shape is None:
            return
        if self.child is not None:
            self.child.shape = shape
        npts = np.prod(shape)
        self.buffer = np.zeros((3, npts), dtype=np.uint8)
        self.index = 0
        self._initialized = False

    @property
    def order(self) -> int:
        return self._order

    @order.setter
    def order(self, order: int) -> None:
        self._order = int(np.clip(order, 0, 10))
        if self._order > 0:
            self.child = VMedian(order=self._order - 1, shape=self.shape)
        self._initialized = False


vmedian = VMedian  # backward-compatibility alias
