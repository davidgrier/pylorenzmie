#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray
from pylorenzmie.lib import (LMObject, Properties)


@dataclass
class Particle(LMObject):

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
    r_0: numpy.ndarray
        3-dimensional origin of coordinate system
    x_0 : float
        x coordinate of origin
    y_0 : float
        y coordinate of origin
    z_0 : float
        z coordinate of origin

    NOTE: The units of coordinates are not specified

    Methods
    -------
    ab(n_m, wavelength) : numpy.ndarray
        Returns Mie scattering coefficients
    '''

    x_p: float = 0.
    y_p: float = 0.
    z_p: float = 100.
    x_0: float = field(repr=False, default=0.)
    y_0: float = field(repr=False, default=0.)
    z_0: float = field(repr=False, default=0.)

    @property
    def r_p(self) -> NDArray[np.float64]:
        '''Three-dimensional coordinates of particle's center'''
        return np.asarray([self.x_p, self.y_p, self.z_p])

    @r_p.setter
    def r_p(self, r_p: NDArray[np.float64]) -> None:
        self.x_p, self.y_p, self.z_p = r_p

    @property
    def r_0(self) -> NDArray[np.float64]:
        '''Three-dimensional coordinates of origin'''
        return np.asarray([self.x_0, self.y_0, self.z_0])

    @r_0.setter
    def r_0(self, r_0: NDArray[np.float64]) -> None:
        self.x_0, self.y_0, self.z_0 = r_0

    @LMObject.properties.getter
    def properties(self) -> Properties:
        return {'x_p': self.x_p,
                'y_p': self.y_p,
                'z_p': self.z_p}

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


def example() -> None:  # pragma: no cover
    p = Particle()
    print(p)
    print(p.r_p)
    p.x_p = 100.
    print(p.r_p)
    print(p.ab())
    print(p.properties)


if __name__ == '__main__':  # pragma: no cover
    example()
