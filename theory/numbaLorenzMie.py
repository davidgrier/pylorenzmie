#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pylorenzmie.theory.LorenzMie import LorenzMie
import numba as nb
import numpy as np


@nb.jit(nopython=True, fastmath=True, cache=True, parallel=True)
def compute_field_jit(kdr, buffers, ab, cartesian, bohren):
    '''Returns the field scattered by the particle at each coordinate

    Arguments
    ----------
    ab : numpy.ndarray
        [2, norders] Mie scattering coefficients

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
    Mo1n, Ne1n, Es, Ec = buffers

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
    kρ = np.hypot(kx, ky)
    kr = np.hypot(kρ, kz)
    sinkr = np.sin(kr)
    coskr = np.cos(kr)

    φ = np.arctan2(ky, kx)
    cosφ = np.cos(φ)
    sinφ = np.sin(φ)
    θ = np.arctan2(kρ, kz)
    cosθ = np.cos(θ)
    sinθ = np.sin(θ)

    # SPECIAL FUNCTIONS
    # starting points for recursive function evaluation ...
    # 1. Riccati-Bessel radial functions, page 478.
    # Particles above the focal plane create diverging waves
    # described by Eq. (4.13) for $h_n^{(1)}(kr)$. These have z > 0.
    # Those below the focal plane appear to be converging from the
    # perspective of the camera. They are described by Eq. (4.14)
    # for $h_n^{(2)}(kr)$, and have z < 0. We can select the
    # appropriate case by applying the correct sign of the imaginary
    # part of the starting functions...
    factor = 1.j * np.sign(kz) if bohren else -1.j * np.sign(kz)
    ξ_nm2 = coskr + factor * sinkr  # \xi_{-1}(kr)
    ξ_nm1 = sinkr - factor * coskr  # \xi_0(kr)

    # 2. Angular functions (4.47), page 95
    # \pi_0(\cos\theta)
    π_nm1 = np.zeros(shape)
    # \pi_1(\cos\theta)
    π_n = np.ones(shape)

    # 3. Vector spherical harmonics: (r, θ, φ)
    Mo1n[0].fill(0.j)            # no radial component

    # storage for scattered field
    Es.fill(0.j)

    # COMPUTE field by summing partial waves
    for n in range(1, norders):
        # upward recurrences ...
        # 4. Legendre factor (4.47)
        # Method described by Wiscombe (1980)

        swisc = π_n * cosθ
        twisc = swisc - π_nm1
        # np.multiply(π_n, cosθ, out=swisc)
        # np.subtract(swisc, π_nm1, out=twisc)
        τ_n = π_nm1 - n * twisc  # -\tau_n(\cos\theta)

        # ... Riccati-Bessel function, page 478
        ξ_n = (2.*n - 1.) * (ξ_nm1 / kr) - ξ_nm2  # \xi_n(kr)

        # ... Deirmendjian's derivative
        Dn = n * (ξ_n / kr) - ξ_nm1

        # vector spherical harmonics (4.50)
        Mo1n[1] = π_n * ξ_n      # ... divided by cosφ/kr
        Mo1n[2] = τ_n * ξ_n      # ... divided by sinφ/kr

        # ... divided by cosφ sinθ/kr^2
        Ne1n[0] = (n*n + n) * π_n * ξ_n
        Ne1n[1] = τ_n * Dn       # ... divided by cosφ/kr
        Ne1n[2] = π_n * Dn       # ... divided by sinφ/kr

        # prefactor, page 93
        En = 1.j**n * (2.*n + 1.) / (n*n + n)

        # the scattered field in spherical coordinates (4.45)
        Es += (1.j * En * ab[n, 0]) * Ne1n
        Es -= (En * ab[n, 1]) * Mo1n

        # upward recurrences ...
        # ... angular functions (4.47)
        # Method described by Wiscombe (1980)
        π_nm1 = π_n
        π_n = swisc + (1. + 1./n) * twisc

        # ... Riccati-Bessel function
        ξ_nm2 = ξ_nm1
        ξ_nm1 = ξ_n
        # n: multipole sum

    # geometric factors were divided out of the vector
    # spherical harmonics for accuracy and efficiency ...
    # ... put them back at the end.
    Es[0] *= cosφ * sinθ / (kr * kr)
    Es[1] *= cosφ / kr
    Es[2] *= sinφ / kr

    # By default, the scattered wave is returned in spherical
    # coordinates.  Project components onto Cartesian coordinates.
    # Assumes that the incident wave propagates along z and
    # is linearly polarized along x
    if cartesian:
        Ec[0] = Es[0] * sinθ * cosφ
        Ec[0] += Es[1] * cosθ * cosφ
        Ec[0] -= Es[2] * sinφ

        Ec[1] = Es[0] * sinθ * sinφ
        Ec[1] += Es[1] * cosθ * sinφ
        Ec[1] += Es[2] * cosφ
        Ec[2] = Es[0] * cosθ - Es[1] * sinθ
        return Ec
    else:
        return Es


class numbaLorenzMie(LorenzMie):

    method: str = 'numba'

    def compute(self,
                ab: LorenzMie.Coefficients,
                cartesian: bool = True,
                bohren: bool = True) -> LorenzMie.Field:
        return compute_field_jit(self.kdr, self.buffers, ab, cartesian, bohren)


if __name__ == '__main__':  # pragma: no cover
    numbaLorenzMie.example()
