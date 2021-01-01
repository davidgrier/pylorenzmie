#!/usr/bin/env python
# -*- coding: utf-8 -*-

from . import GeneralizedLorenzMie
from .Sphere import Sphere


class LorenzMie(GeneralizedLorenzMie):

    '''
    NOTE: DEPRECATED: This class is made redundant by
    the default settings of GLM.

    Class to compute light scattering by a sphere

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
        super(LorenzMie, self).__init__(**kwargs)
        if particle is None:
            self.particle = Sphere()
            if a_p is not None:
                self.particle.a_p = a_p
            if n_p is not None:
                self.particle.n_p = n_p
            if r_p is not None:
                self.particle.r_p = r_p
        else:
            self.particle = particle


if __name__ == '__main__':
    try:
        import cupy as np
        using_cupy = True
    except:
        import numpy as np
        using_cupy = False
    from matplotlib import pyplot as plt
    from pylorenzmie.theory.Instrument import Instrument, coordinates

    particle = Sphere()
    particle.r_p = [150, 150, 200]
    particle.a_p = 0.5
    particle.n_p = 1.45
    particle2 = Sphere()
    particles = [particle, particle2]
    # Form image with default instrument
    instrument = Instrument()
    instrument.magnification = 0.135
    instrument.wavelength = 0.447
    instrument.n_m = 1.335
    # Use Generalized Lorenz-Mie theory to compute field
    kernel = LorenzMie(particle=particles)
    kernel.instrument = instrument
    kernel.coordinates = coordinates((201, 201))
    field = kernel.field()
    # Compute hologram from field and show it
    field[0, :] += 1.
    hologram = np.sum(np.real(field * np.conj(field)), axis=0)
    if using_cupy:
        hologram = hologram.get()
    plt.imshow(hologram.reshape(201, 201), cmap='gray')
    plt.show()
