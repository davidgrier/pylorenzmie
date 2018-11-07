import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import pycuda.cumath as cumath
import numpy as np


def cuda_field(ab, krv, cartesian=True, bohren=True):
    '''Returns the field scattered by the particle at each coordinate

    Parameters
    ----------
    ab : numpy.ndarray
        Mie scattering coefficients
    krv : numpy.ndarray
        Reduced vector displacements of particle from image coordinates
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

    ab = ab.astype(np.complex64)
    nc = ab.shape[0]  # number of partial waves in sum

    # GEOMETRY
    # 1. particle displacement [pixel]
    # Note: The sign convention used here is appropriate
    # for illumination propagating in the -z direction.
    # This means that a particle forming an image in the
    # focal plane (z = 0) is located at positive z.
    # Accounting for this by flipping the axial coordinate
    # is equivalent to using a mirrored (left-handed)
    # coordinate system.
    kx = gpuarray.to_gpu(krv[:, 0]).astype(np.float32)
    ky = gpuarray.to_gpu(krv[:, 1]).astype(np.float32)
    kz = gpuarray.to_gpu(-krv[:, 2]).astype(np.float32)
    npts = len(kx)

    # 2. geometric factors
    krho = cumath.sqrt(kx * kx + ky * ky)
    cosphi = kx / krho
    sinphi = ky / krho

    kr = cumath.sqrt(krho * krho + kz * kz)
    costheta = kz / kr
    sintheta = krho / kr

    sinkr = cumath.sin(kr)
    coskr = cumath.cos(kr)

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
    factor = 1.j * kz / abs(kz)
    if not bohren:
        factor *= -1.
    xi_nm2 = coskr + factor * sinkr  # \xi_{-1}(kr)
    xi_nm1 = sinkr - factor * coskr  # \xi_0(kr)

    # 2. Angular functions (4.47), page 95
    pi_nm1 = 0.                      # \pi_0(\cos\theta)
    pi_n = 1.                        # \pi_1(\cos\theta)

    # 3. Vector spherical harmonics: [r,theta,phi]
    mo1n = gpuarray.zeros([3, npts], dtype=np.complex64)
    ne1n = gpuarray.empty([3, npts], dtype=np.complex64)

    # storage for scattered field
    es = gpuarray.zeros([3, npts], dtype=np.complex64)

    # COMPUTE field by summing partial waves
    for n in range(1, nc):
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
        # mo1n[0, :] = 0.j           # no radial component
        mo1n[1, :] = pi_n * xi_n     # ... divided by cosphi/kr
        mo1n[2, :] = tau_n * xi_n    # ... divided by sinphi/kr

        # ... divided by cosphi sintheta/kr^2
        ne1n[0, :] = n * (n + 1.) * pi_n * xi_n
        ne1n[1, :] = tau_n * dn      # ... divided by cosphi/kr
        ne1n[2, :] = pi_n * dn       # ... divided by sinphi/kr

        # prefactor, page 93
        en = 1.j**n * (2. * n + 1.) / n / (n + 1.)

        # the scattered field in spherical coordinates (4.45)
        es += np.complex64(1.j * en * ab[n, 0]) * ne1n
        es -= np.complex64(en * ab[n, 1]) * mo1n

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
        ec = gpuarray.empty_like(es)

        ec[0, :] = es[0, :] * sintheta * cosphi
        ec[0, :] += es[1, :] * costheta * cosphi
        ec[0, :] -= es[2, :] * sinphi

        ec[1, :] = es[0, :] * sintheta * sinphi
        ec[1, :] += es[1, :] * costheta * sinphi
        ec[1, :] += es[2, :] * cosphi

        ec[2, :] = es[0, :] * costheta - es[1, :] * sintheta
        return ec.get()
    else:
        return es.get()
