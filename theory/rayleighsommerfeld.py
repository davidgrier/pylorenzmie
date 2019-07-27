# -*- coding: utf-8 -*-

import numpy as np


def hanning(nx, ny):
    """
    Calculates the Hanning Window of size (nx,ny)
    """
    if (nx <= 0):
        raise ValueError('nx must be greater than zero')
    if (ny < 0):
        raise ValueError('ny cannot be negative')

    xwindow = np.hanning(nx)
    if ny > 0:
        ywindow = np.hanning(ny)
        return np.sqrt(np.outer(xwindow, ywindow))
    else:
        return xwindow


def rayleighsommerfeld(a, z,
                       wavelength=0.447,
                       magnification=0.135,
                       nozphase=False,
                       hanning=False):
    """
    Compute electric fields propagated by a distance or
    set of distances above the imaging plane via
    Rayleigh-Sommerfeld approximation.

    Args:
        a: A two dimensional intensity array.
        z: displacement(s) from the focal plane [pixels].

    Keywords:
        wavelength: Wavelength of light in medium [micrometers].
            Default: 0.447
        magnification: Micrometers per pixel.
            Default: 0.135

    Returns:
        Complex electric fields at a plane or set of planes z.
    """

    if a.ndim != 2:
        raise ValueError('a must be a two-dimensional hologram')
    a = np.array(a, dtype=complex)
    ny, nx = a.shape

    z = np.atleast_1d(z)
    nz = len(z)

    # important factors
    k = 2.*np.pi * magnification/wavelength  # wavenumber [radians/pixel]

    # phase factor for Rayleigh-Sommerfeld propagator in Fourier space
    # Compute factor k*sqrt(1-qx**2+qy**2)
    # (FIXME MDH): Do I need to neglect the endpoint?
    qx = np.linspace(-0.5, 0.5, nx, endpoint=False, dtype=complex)
    qy = np.linspace(-0.5, 0.5, ny, endpoint=False, dtype=complex)
    qx, qy = np.meshgrid(qx, qy)
    qsq = qx**2 + qy**2
    qsq *= (wavelength/magnification)**2

    qfactor = k * np.sqrt(1. - qsq)

    if nozphase:
        qfactor -= k

    if hanning:
        qfactor *= hanning(ny, nx)

    # Account for propagation and absorption
    ikappa = 1j * np.real(qfactor)
    gamma = np.imag(qfactor)

    # Go to Fourier space and apply RS propagation operator
    E = np.fft.ifft2(a - 1.)                   # avg(a-1) should = 0.
    E = np.fft.fftshift(E)
    res = np.zeros([ny, nx, nz], dtype=complex)
    for n in range(0, nz):
        Hqz = np.exp((ikappa * z[n] - gamma * abs(z[n])))
        thisE = E * Hqz                        # convolve with propagator
        thisE = np.fft.ifftshift(thisE)        # shift center
        thisE = np.fft.fft2(thisE)             # transform back to real space
        res[:, :, n] = thisE                   # save result

    return res + 1.  # undo the previous reduction by 1.
