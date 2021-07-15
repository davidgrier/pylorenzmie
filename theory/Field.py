# -*- coding: utf-8 -*-

import numpy as np

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class Field(object):
    '''Base class for computed fields'''

    def __init__(self,
                 coordinates=None,
                 **kwargs):
        self.coordinates = coordinates

    @property
    def coordinates(self):
        '''Three-dimensional coordinates at which field is calculated

        Expected shape is (3, npts)
        '''
        return self._coordinates

    @coordinates.setter
    def coordinates(self, coordinates):
        if coordinates is None:
            self._coordinates = None
            return
        c = np.array(coordinates)
        if c.ndim == 1:          # only x specified
            c = np.vstack((c, np.zeros((2, c.size))))
        elif c.shape[0] == 2:    # only (x, y) specified
            c = np.vstack((c, np.zeros(c.shape[1])))
        if c.shape[0] != 3:      # pragma: no cover
            raise ValueError(
                'coordinates should have shape ({1|2|3}, npts).')
        self._coordinates = c
