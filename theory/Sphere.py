#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pylorenzmie.theory.Particle import Particle
import numpy as np
from numba import jit

'''
REFERENCES
1. Adapted from Chapter 8 in
   C. F. Bohren and D. R. Huffman,
   Absorption and Scattering of Light by Small Particles,
   (New York, Wiley, 1983).

2. W. Yang,
   Improved recursive algorithm for light scattering
   by a multilayered sphere,
   Applied Optics 42, 1710--1720 (2003).

3. O. Pena, U. Pal,
   Scattering of electromagnetic radiation by a multilayered sphere,
   Computer Physics Communications 180, 2348--2354 (2009).
   NB: Equation numbering follows this reference.

4. W. J. Wiscombe,
   Improved Mie scattering algorithms,
   Applied Optics 19, 1505-1509 (1980).

5. A. A. R. Neves and D. Pisignano,
   Effect of finite terms on the truncation error of Mie series,
   Optics Letters 37, 2481-2420 (2012).

HISTORY
Adapted from the IDL routine sphere_coefficients.pro
which calculates scattering coefficients for layered spheres.
The IDL code was
Copyright (c) 2010-2016 F. C. Cheong and D. G. Grier
The present python adaptation is
Copyright (c) 2018 Mark D. Hannel and David G. Grier
'''


def mie_coefficients(a_p, n_p, k_p, n_m, wavelength):
    '''Calculate the Mie scattering coefficients for a sphere

    This works for a (multilayered) sphere illuminated by
    a coherent plane wave that is linearly polarized in the
    x direction.

    Parameters
    ----------
    a_p : float or numpy.ndarray
        radii of the layers in the sphere [um]
    n_p : float or numpy.ndarray
        Refractive indexes of sphere's layers
    k_p : float or numpy.ndarray
        Absorption coefficient of sphere's layers
    n_m : complex
        (complex) refractive index of medium
    wavelength : float
        wavelength of light [um]

    Returns
    -------
    ab : numpy.ndarray
        Mie AB coefficients
    '''
    a_p = np.atleast_1d(np.asarray(a_p))
    n_p = np.atleast_1d(np.asarray(n_p))
    k_p = np.atleast_1d(np.asarray(k_p))

    # size parameters for layers
    k = 2.*np.pi/wavelength     # wave number in vacuum [um^-1]
    k *= np.real(n_m)           # wave number in medium
    x = k * a_p                 # size parameter in each layer
    m = (n_p + 1.j*k_p) / n_m   # relative refractive index in each layer

    # number of terms in partial-wave expansion
    nmax = wiscombe_yang(x, m)

    # storage for results
    ab = np.empty([nmax+1, 2], complex)
    d1_z1 = np.empty(nmax+1, complex)
    d1_z2 = np.empty(nmax+1, complex)
    d3_z1 = np.empty(nmax+1, complex)
    d3_z2 = np.empty(nmax+1, complex)
    psi = np.empty(nmax+1, complex)
    zeta = np.empty(nmax+1, complex)
    q = np.empty(nmax+1, complex)
    return _mie_coefficients(a_p, n_p, k_p, n_m, wavelength,
                             x, m, nmax,
                             ab, d1_z1, d1_z2,
                             d3_z1, d3_z2, psi, zeta, q)


@jit(nopython=True)
def wiscombe_yang(x, m):
    '''Return the number of terms to keep in partial wave expansion

    Equation numbers refer to Wiscombe (1980) and Yang (2003).

    Parameters
    ----------
    x : complex or numpy.ndarray
        size parameters for each layer
    m : complex or numpy.ndarray
        relative refractive indexes of the layers

    Returns
    -------
    ns : int
        Number of terms to retain in the partial-wave expansion
    '''

    # Wiscombe (1980)
    xl = np.abs(x[-1])
    if xl <= 8.:
        ns = np.floor(xl + 4. * xl**(1. / 3.) + 1.)
    elif xl <= 4200.:
        ns = np.floor(xl + 4.05 * xl**(1. / 3.) + 2.)
    else:
        ns = np.floor(xl + 4. * xl**(1. / 3.) + 2.)

    # Yang (2003) Eq. (30)
    xm = np.abs(x * m)
    xm_1 = np.abs(np.roll(x, -1) * m)
    nstop = max(ns, xm.max(), xm_1.max())
    return int(nstop)


@jit(nopython=True)
def _mie_coefficients(a_p, n_p, k_p, n_m, wavelength, x, m, nmax,
                      ab, d1_z1, d1_z2, d3_z1, d3_z2, psi, zeta, q):
    # initialization
    d1_z1[nmax] = 0.                                          # Eq. (16a)
    d1_z2[nmax] = 0.
    d3_z1[0] = 1.j                                            # Eq. (18b)
    d3_z2[0] = 1.j

    # iterate outward from the sphere's core
    nlayers = a_p.size
    z1 = x[0] * m[0]
    for n in range(nmax, 0, -1):
        d1_z1[n-1] = n/z1 - 1./(d1_z1[n] + n/z1)              # Eq. (16b)
    ha = d1_z1.copy()                                         # Eq. (7a)
    hb = d1_z1.copy()                                         # Eq. (8a)

    # iterate outward from layer 2 to layer L
    # account for 0-based python indexing (ii = l-1)
    for ii in range(1, nlayers):
        z1 = m[ii] * x[ii]
        z2 = m[ii] * x[ii-1]

        # downward recurrence for D1 (D1[nmax] = 0)
        for n in range(nmax, 0, -1):
            d1_z1[n-1] = n/z1 - 1./(d1_z1[n] + n/z1)          # Eq. (16b)
            d1_z2[n-1] = n/z2 - 1./(d1_z2[n] + n/z2)

        # upward recurrence for PsiZeta, D3, Q
        a1, b1 = np.real(z1), np.imag(z1)
        a2, b2 = np.real(z2), np.imag(z2)
        psizeta_z1 = 0.5*(1.-np.exp(2.j*a1)*np.exp(-2.*b1))   # Eq. (18a)
        psizeta_z2 = 0.5*(1.-np.exp(2.j*a2)*np.exp(-2.*b2))   # Eq. (18a)
        q[0] = ((np.exp(-2.j*a1) - np.exp(-2.*b1)) /
                (np.exp(-2.j*a2) - np.exp(-2.*b2))) * \
            np.exp(-2.*(b2 - b1))                             # Eq. (19a)
        for n in range(1, nmax+1):
            psizeta_z1 *= (n/z1 - d1_z1[n-1]) * \
                          (n/z1 - d3_z1[n-1])                 # Eq. (18c)
            psizeta_z2 *= (n/z2 - d1_z2[n-1]) * \
                          (n/z2 - d3_z2[n-1])
            d3_z1[n] = d1_z1[n] + 1.j/psizeta_z1              # Eq. (18d)
            d3_z2[n] = d1_z2[n] + 1.j/psizeta_z2
            q[n] = q[n-1] * (x[ii-1]/x[ii])**2 * \
                (z2 * d1_z2[n] + n)/(z1 * d1_z1[n] + n) * \
                (n - z2 * d3_z2[n-1])/(n - z1 * d3_z1[n-1])   # Eq. (19b)
        # update Ha and Hb
        g1 = m[ii] * ha - m[ii-1] * d1_z2                     # Eq. (12)
        g2 = m[ii] * ha - m[ii-1] * d3_z2                     # Eq. (13)
        ha = (g2 * d1_z1 - q * g1 * d3_z1) / (g2 - q * g1)    # Eq. (7b)

        g1 = m[ii-1] * hb - m[ii] * d1_z2                     # Eq. (14)
        g2 = m[ii-1] * hb - m[ii] * d3_z2                     # Eq. (15)
        hb = (g2 * d1_z1 - q * g1 * d3_z1) / (g2 - q * g1)    # Eq. (8b)
    # ii (layers)

    # iterate into medium (m = 1.)
    z1 = x[-1]
    # downward recurrence for D1 (D1[nmax] = 0)
    for n in range(nmax, 0, -1):
        d1_z1[n-1] = n/z1 - (1./(d1_z1[n] + n/z1))            # Eq. (16b)

    # upward recurrence for Psi, Zeta, PsiZeta and D3
    psi[0] = np.sin(z1)                                       # Eq. (20a)
    zeta[0] = -1.j * np.exp(1.j * z1)                         # Eq. (21a)
    psizeta = 0.5 * (1. - np.exp(2.j * z1))                   # Eq. (18a)
    for n in range(1, nmax+1):
        psi[n] = psi[n-1] * (n/z1 - d1_z1[n-1])               # Eq. (20b)
        zeta[n] = zeta[n-1] * (n/z1 - d3_z1[n-1])             # Eq. (21b)
        psizeta *= (n/z1 - d1_z1[n-1]) * (n/z1 - d3_z1[n-1])  # Eq. (18c)
        d3_z1[n] = d1_z1[n] + 1.j/psizeta                     # Eq. (18d)

    # Scattering coefficients
    n = np.arange(nmax+1)
    fac = ha/m[-1] + n/x[-1]
    ab[:, 0] = (fac * psi - np.roll(psi, 1)) / \
        (fac * zeta - np.roll(zeta, 1))                 # Eq. (5)
    fac = hb*m[-1] + n/x[-1]
    ab[:, 1] = (fac * psi - np.roll(psi, 1)) / \
        (fac * zeta - np.roll(zeta, 1))                 # Eq. (6)
    ab[0, :] = complex(0., 0.)

    return ab


class Sphere(Particle):

    '''
    Abstraction of a spherical particle for Lorenz-Mie micrsocopy

    ...

    Attributes
    ----------
    a_p : float or numpy.ndarray
        radius of particle [um]
        or array containing radii of concentric shells
    n_p : float or numpy.ndarray
        refractive index of particle
        or array containing refractive indexes of shells
    k_p : float or numpy.ndarray
        absorption coefficient of particle
        or array containing absorption coefficients of shells

    Methods
    -------
    ab(n_m, wavelength) : numpy.ndarray
        returns the Mie scattering coefficients for the sphere
    '''

    def __init__(self,
                 a_p=1.,   # radius of sphere [um]
                 n_p=1.5,  # refractive index of sphere
                 k_p=0.,   # absorption coefficient
                 **kwargs):
        super(Sphere, self).__init__(**kwargs)
        self.a_p = a_p
        self.n_p = n_p
        self.k_p = k_p
        self.ab(1.339, 0.447)  # call dummy method for jit compilation

    def __str__(self):
        name = self.__class__.__name__
        str = '{}(a_p={}, n_p={}, k_p={}, r_p={})'
        return str.format(name, self.a_p, self.n_p, self.k_p, self.r_p)

    @property
    def a_p(self):
        '''Radius of sphere [um]'''
        if self._a_p.size == 1:
            return np.asscalar(self._a_p)
        else:
            return self._a_p

    @a_p.setter
    def a_p(self, a_p):
        self._a_p = np.asarray(a_p, dtype=float)

    @property
    def n_p(self):
        '''Refractive index of sphere'''
        if self._n_p.size == 1:
            return np.asscalar(self._n_p)
        else:
            return self._n_p

    @n_p.setter
    def n_p(self, n_p):
        self._n_p = np.asarray(n_p, dtype=float)

    @property
    def k_p(self):
        '''Absorption coefficient of sphere'''
        if self._k_p.size == 1:
            return np.asscalar(self._k_p)
        else:
            return self._k_p

    @k_p.setter
    def k_p(self, k_p):
        self._k_p = np.asarray(k_p, dtype=float)

    @property
    def properties(self):
        props = Particle.properties.fget(self)
        props['a_p'] = self.a_p
        props['n_p'] = self.n_p
        props['k_p'] = self.k_p
        return props

    @properties.setter
    def properties(self, properties):
        Particle.properties.fset(self, properties)

    def ab(self, n_m, wavelength):
        '''Returns the Mie scattering coefficients

        Parameters
        ----------
        n_m : complex
            Refractive index of medium
        wavelength : float
            Vacuum wavelength of light [um]

        Returns
        -------
        ab : numpy.ndarray
            Mie AB scattering coefficients
        '''
        return mie_coefficients(self.a_p, self.n_p, self.k_p, n_m,
                                wavelength)


if __name__ == '__main__':
    from time import time
    s = Sphere(a_p=0.75, n_p=1.5)
    print(s.a_p, s.n_p)
    start = time()
    s.ab(1.339, 0.447)
    print(time()-start)
