#!/usr/bin/env python
# -*- coding: utf-8 -*-

from numba import njit
from dataclasses import dataclass
from pylorenzmie.theory.LorenzMie import (LorenzMie, example)
from typing import (Optional, List)
import numpy as np

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

np.seterr(all='raise')


@dataclass
class numbaLorenzMie(LorenzMie):

    method: str = 'numba'

    @staticmethod
    @njit(cache=True)
    def pad(coordinates: Optional[np.ndarray]) -> None:
        logger.debug('Setting coordinates')
        c = np.atleast_2d(0. if coordinates is None else coordinates)
        ndim, npts = c.shape
        if ndim > 3:
            raise ValueError(f'Incompatible shape: {coordinates.shape=}')
        return np.vstack([c, np.zeros((3-ndim, npts))])

    @staticmethod
    @njit(cache=True)
    def compute(ab: np.ndarray,
                kdr: np.ndarray,
                buffers: List[np.ndarray],
                cartesian: bool = True,
                bohren: bool = True) -> np.ndarray:  # pragma: no cover
        '''Returns the field scattered by the particle at each coordinate

        Arguments
        ----------
        ab : numpy.ndarray
            [2, norders] Mie scattering coefficients
        kdr : numpy.ndarray
            [3, npts] Coordinates at which field is evaluated
            relative to the center of the scatterer. Coordinates
            are assumed to be multiplied by the wavenumber of
            light in the medium, and so are dimensionless.

        Keywords
        --------
        cartesian : bool
            If set, return field projected onto Cartesian coordinates.
            Otherwise, return polar projection.
        bohren : bool
            If set, use sign convention from Bohren and Huffman.
            Otherwise, use opposite sign convention.

        Returns
        -------
        field : numpy.ndarray
            [3, npts] array of complex vector values of the
            scattered field at each coordinate.
        '''

        norders = ab.shape[0]  # number of partial waves in sum
        mo1n, ne1n, es, ec = buffers

        # GEOMETRY
        # 1. particle displacement [pixel]
        # Note: The sign convention used here is appropriate
        # for illumination propagating in the -z direction.
        # This means that a particle forming an image in the
        # focal plane (z = 0) is located at positive z.
        # Accounting for this by flipping the axial coordinate
        # is equivalent to using a mirrored (left-handed)
        # coordinate system.
        kx = kdr[0, :]
        ky = kdr[1, :]
        kz = -kdr[2, :]
        shape = kx.shape

        # 2. geometric factors
        krho = np.hypot(kx, ky)
        kr = np.hypot(krho, kz)

        phi = np.arctan2(ky, kx)
        cosphi = np.cos(phi)
        sinphi = np.sin(phi)
        theta = np.arctan2(krho, kz)
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        sinkr = np.sin(kr)
        coskr = np.cos(kr)

        # SPECIAL FUNCTIONS
        # starting points for recursive function evaluation ...
        # 1. Riccati-Bessel radial functions, page 478.
        # Particles above the focal plane create diverging waves
        # described by Eq. (4.13) for $h_n^{(1)}(kr)$. These have z > 0.
        # Those below the focal plane appear to be converging from the
        # perspective of the camera. They are descrinbed by Eq. (4.14)
        # for $h_n^{(2)}(kr)$, and have z < 0. We can select the
        # appropriate case by applying the correct sign of the imaginary
        # part of the starting functions...
        factor = 1.j * np.sign(kz) if bohren else -1.j * np.sign(kz)

        xi_nm2 = coskr + factor * sinkr  # \xi_{-1}(kr)
        xi_nm1 = sinkr - factor * coskr  # \xi_0(kr)

        # 2. Angular functions (4.47), page 95
        # \pi_0(\cos\theta)
        pi_nm1 = np.zeros(shape)
        # \pi_1(\cos\theta)
        pi_n = np.ones(shape)

        # 3. Vector spherical harmonics: [r,theta,phi]
        mo1n[0, :] = 0.j                 # no radial component

        # storage for scattered field
        es.fill(0.j)

        # COMPUTE field by summing partial waves
        for n in range(1, norders):
            # upward recurrences ...
            # 4. Legendre factor (4.47)
            # Method described by Wiscombe (1980)

            swisc = pi_n * costheta
            twisc = swisc - pi_nm1
            tau_n = pi_nm1 - n * twisc  # -\tau_n(\cos\theta)

            # ... Riccati-Bessel function, page 478
            xi_n = (2. * n - 1.) * (xi_nm1 / kr) - xi_nm2  # \xi_n(kr)

            # ... Deirmendjian's derivative
            dn = (n * xi_n) / kr - xi_nm1

            # vector spherical harmonics (4.50)
            mo1n[1, :] = pi_n * xi_n     # ... divided by cosphi/kr
            mo1n[2, :] = tau_n * xi_n    # ... divided by sinphi/kr

            # ... divided by cosphi sintheta/kr^2
            ne1n[0, :] = n * (n + 1.) * pi_n * xi_n
            ne1n[1, :] = tau_n * dn      # ... divided by cosphi/kr
            ne1n[2, :] = pi_n * dn       # ... divided by sinphi/kr

            # prefactor, page 93
            en = 1.j**n * (2. * n + 1.) / n / (n + 1.)

            # the scattered field in spherical coordinates (4.45)
            es += (1.j * en * ab[n, 0]) * ne1n
            es -= (en * ab[n, 1]) * mo1n

            # upward recurrences ...
            # ... angular functions (4.47)
            # Method described by Wiscombe (1980)
            pi_nm1 = pi_n
            pi_n = swisc + ((n + 1.) / n) * twisc

            # ... Riccati-Bessel function
            xi_nm2 = xi_nm1
            xi_nm1 = xi_n
            # n: multipole sum

        # geometric factors were divided out of the vector
        # spherical harmonics for accuracy and efficiency ...
        # ... put them back at the end.
        radialfactor = 1. / kr
        es[0, :] *= cosphi * sintheta * radialfactor**2
        es[1, :] *= cosphi * radialfactor
        es[2, :] *= sinphi * radialfactor

        # By default, the scattered wave is returned in spherical
        # coordinates.  Project components onto Cartesian coordinates.
        # Assumes that the incident wave propagates along z and
        # is linearly polarized along x

        if cartesian:
            ec[0, :] = es[0, :] * sintheta * cosphi
            ec[0, :] += es[1, :] * costheta * cosphi
            ec[0, :] -= es[2, :] * sinphi

            ec[1, :] = es[0, :] * sintheta * sinphi
            ec[1, :] += es[1, :] * costheta * sinphi
            ec[1, :] += es[2, :] * cosphi
            ec[2, :] = (es[0, :] * costheta -
                        es[1, :] * sintheta)
            return ec
        else:
            return es


if __name__ == '__main__':  # pragma: no cover
    example(numbaLorenzMie)
