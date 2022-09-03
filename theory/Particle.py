#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataclasses import dataclass
import numpy as np
import json
from typing import Optional, Any


@dataclass
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

    x_p: float = 0.
    y_p: float = 0.
    z_p: float = 100.

    @property
    def r_p(self) -> np.ndarray:
        '''Three-dimensional coordinates of particle's center'''
        return np.asarray([self.x_p, self.y_p, self.z_p])

    @r_p.setter
    def r_p(self, r_p: np.ndarray) -> None:
        self.x_p, self.y_p, self.z_p = r_p

    @property
    def properties(self) -> dict:
        return dict(x_p=self.x_p, y_p=self.y_p, z_p=self.z_p)

    @properties.setter
    def properties(self, properties: dict) -> None:
        for name, value in properties.items():
            if hasattr(self, name):
                setattr(self, name, value)

    def dumps(self, **kwargs: Optional[Any]) -> str:
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

    def loads(self, s: str) -> None:
        '''Loads JSON string of adjustable properties

        Parameters
        ----------
        s : str
            JSON-encoded string of properties
        '''
        self.properties = json.loads(s)

    def ab(self,
           n_m: complex = 1.+0.j,
           wavelength: float = 0.) -> np.ndarray:
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


if __name__ == '__main__':  # pragma: no cover
    p = Particle()
    print(p.r_p)
    p.x_p = 100.
    print(p.r_p)
    print(p.ab())
