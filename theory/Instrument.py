# /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import json
import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class Instrument(object):
    '''
    Abstraction of an in-line holographic microscope

    The instrument forms an image of the light scattering
    pattern for Lorenz-Mie microscopy

    ...

    Properties
    ----------
    wavelength : float
        Vacuum wavelength of light [um]
    magnification : float
        Effective size of pixels [um/pixel]
    n_m : float
        Refractive index of medium
    background : float or numpy.ndarray
        Background image
    noise : float
        Estimated noise as a percentage of the mean value
    dark_count : float
        Dark count of camera
    properties : dict
        Adjustable properties of the instrument model

    Methods
    -------
    wavenumber(in_medium=True, magnified=True) : float
        Wavenumber of light
    '''

    def __init__(self,
                 wavelength=0.532,
                 magnification=0.135,
                 n_m=1.335,
                 background=1.,
                 noise=0.05,
                 dark_count=0.,
                 **kwargs):
        self.wavelength = wavelength
        self.magnification = magnification
        self.n_m = n_m
        self.noise = noise
        self.dark_count = dark_count
        self.background = background

    def __str__(self):
        fmt = '<{}(wavelength={}, magnification={}, n_m={})>'
        return fmt.format(self.__class__.__name__,
                          self.wavelength,
                          self.magnification,
                          self.n_m)

    def __repr__(self):
        return self.__str__()

    @property
    def wavelength(self):
        '''Wavelength of light in vacuum [um]'''
        return self._wavelength

    @wavelength.setter
    def wavelength(self, wavelength):
        logger.debug('wavelength: {} [um]'.format(wavelength))
        self._wavelength = float(wavelength)

    @property
    def magnification(self):
        '''Magnification of microscope [um/pixel]'''
        return self._magnification

    @magnification.setter
    def magnification(self, magnification):
        logger.debug('magnification: {} [um/pixel]'.format(magnification))
        self._magnification = float(magnification)

    @property
    def n_m(self):
        '''Complex refractive index of medium'''
        return self._n_m

    @n_m.setter
    def n_m(self, n_m):
        logger.debug('medium index: {}'.format(n_m))
        self._n_m = float(n_m)

    @property
    def background(self):
        '''Background image'''
        return self._background

    @background.setter
    def background(self, background):
        self._background = background

    @property
    def dark_count(self):
        '''Dark count of camera'''
        return self._dark_count

    @dark_count.setter
    def dark_count(self, dark_count):
        assert dark_count >= 0, 'dark count is non-negative'
        self._dark_count = dark_count

    @property
    def properties(self):
        props = dict(n_m=self.n_m,
                     wavelength=self.wavelength,
                     magnification=self.magnification)
        return props

    @properties.setter
    def properties(self, properties):
        for property, value in properties.items():
            if hasattr(self, property):
                setattr(self, property, value)

    def dumps(self, **kwargs):
        '''Returns JSON string of adjustable properties

        Parameters
        ----------
        Accepts all keywords of json.dumps()

        Returns
        -------
        str : string
            JSON-encoded string of properties
        '''
        return json.dumps(self.properties, **kwargs)

    def loads(self, str):
        '''Loads JSON strong of adjustable properties

        Parameters
        ----------
        str : string
            JSON-encoded string of properties
        '''
        self.properties = json.loads(str)

    def wavenumber(self, in_medium=True, magnified=True):
        '''Return the wave number of light

        Parameters
        ----------
        in_medium : bool
            If set (default) return the wave number in the medium
            Otherwise, return the wave number in vacuum
        magnified : bool
            If set (default) return the scaled value [radian/pixel]
            Otherwise, return SI value [radian/um]

        Returns
        -------
        k : float
            Wave number
        '''
        k = 2. * np.pi / self.wavelength  # wave number in vacuum
        if in_medium:
            k *= self.n_m                 # ... in medium
        if magnified:
            k *= self.magnification       # ... in image units
        return k


if __name__ == '__main__': # pragma: no cover
    a = Instrument()
    print(a.wavelength, a.magnification)
