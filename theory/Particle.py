#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import json


class Particle(object):

    '''
    Abstraction of a particle for Lorenz-Mie microscopy

    ...

    Attributes
    ----------
    r_p : numpy.ndarray
        3-dimensional coordinates of particle's center
    x_p : float
        x coordinate
    y_p : float
        y coordinate
    z_p : float
        z coordinate
    properties : dict
        dictionary of adjustable properties

    Methods
    -------
    ab(n_m, wavelength) : numpy.ndarray
        Returns the Mie scattering coefficients
    '''

    def __init__(self, r_p=[0, 0, 100], **kwargs):
        '''
        Parameters
        ----------
        r_p : list or numpy.ndarray
            [x, y, z] coordinates of the center of the particle.
        '''
        self.r_p = r_p

    def __str__(self):
        str = '{}(r_p={})'
        r_p = ['{:.2f}'.format(c) for c in self.r_p]
        return str.format(self.__class__.__name__, r_p)

    def __repr__(self):
        return self.__str__()

    @property
    def r_p(self):
        '''Three-dimensional coordinates of particle's center'''
        return self._r_p

    @r_p.setter
    def r_p(self, r_p):
        self._r_p = np.asarray(r_p, dtype=float)

    @property
    def x_p(self):
        return self._r_p[0]

    @x_p.setter
    def x_p(self, x_p):
        self._r_p[0] = float(x_p)

    @property
    def y_p(self):
        return self._r_p[1]

    @y_p.setter
    def y_p(self, y_p):
        self._r_p[1] = float(y_p)

    @property
    def z_p(self):
        return self._r_p[2]

    @z_p.setter
    def z_p(self, z_p):
        self._r_p[2] = float(z_p)

    @property
    def properties(self):
        properties = dict(x_p=self.x_p,
                          y_p=self.y_p,
                          z_p=self.z_p)
        return properties

    @properties.setter
    def properties(self, properties):
        for name, value in properties.items():
            if hasattr(self, name):
                setattr(self, name, value)

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

    def ab(self, n_m=1.+0.j, wavelength=0.):
        '''Returns the Mie scattering coefficients

        Subclasses of Particle should override this
        method.

        Parameters
        ----------
        n_m : complex
            Refractive index of medium
        wavelength: float
            Vacuum wavelength of light [um]

        Returns
        -------
        ab : numpy.ndarray
            Mie AB scattering coefficients
        '''
        return np.asarray([1, 1], dtype=complex)


if __name__ == '__main__': # pragma: no cover
    p = Particle()
    print(p.r_p)
    p.x_p = 100.
    print(p.r_p)
    print(p.ab())
