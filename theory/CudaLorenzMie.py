#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pylorenzmie.theory.CudaGeneralizedLorenzMie import CudaGeneralizedLorenzMie
from pylorenzmie.theory.Sphere import Sphere


class CudaLorenzMie(CudaGeneralizedLorenzMie):

    '''
    Class to compute light scattering by a sphere with CUDA acceleration

    Presumes that the sphere is illumianted by light
    propagating along -z with polarization along x.

    ...

    Attributes
    ----------
    a_p : float or numpy.ndarray, optional
        Starting radius of sphere.  Alternatively, can be
        an array of radii of concentric shells within the sphere
    n_p : complex or numpy.ndarray, optional
        Starting refractive index of sphere.  Alternatively,
        can be an array of refractive indexes of the shells.
    r_p : numpy.ndarray or list, optional
        coordinates of sphere center: (x_p, y_p, z_p)

    Note: After initialization, these attributes can be
    obtained and changed through the object's particle attribute.
    '''

    def __init__(self,
                 a_p=None,
                 n_p=None,
                 r_p=None,
                 particle=None,
                 **kwargs):
        '''
        Parameters
        ----------
        a_p : float or numpy.ndarray, optional
            Starting radius of sphere.  Alternatively, can be
            an array of radii of concentric shells within the sphere
        n_p : complex or numpy.ndarray, optional
            Starting refractive index of sphere.  Alternatively,
            can be an array of refractive indexes of the shells.
        r_p : numpy.ndarray or list, optional
            coordinates of sphere center: (x_p, y_p, z_p)
        '''
        super(CudaLorenzMie, self).__init__(**kwargs)
        self.particle = Sphere()
        if a_p is not None:
            self.particle.a_p = a_p
        if n_p is not None:
            self.particle.n_p = n_p
        if r_p is not None:
            self.particle.r_p = r_p
