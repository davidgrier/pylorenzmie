import numpy as np
import math
from numba import njit, prange
from pylorenzmie.theory.Sphere import mie_coefficients

# See https://llvm.org/docs/LangRef.html#fast-math-flags
# for a list of fastmath flags for LLVM compiler
safe_flags = {'nnan', 'ninf', 'arcp', 'nsz'}


fast_mie_coefficients = njit(mie_coefficients, cache=True)


@njit(parallel=True, fastmath=False, cache=True)
def fastfield(coordinates, r_p, k, phase,
              ab, result, bohren, cartesian):
    '''Returns the field scattered by the particle at each coordinate

    Parameters
    ----------
    ab : numpy.ndarray
        [2, norders] Mie scattering coefficients
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
    length = coordinates.shape[1]

    norders = ab.shape[0]  # number of partial waves in sum

    # GEOMETRY
    # 1. particle displacement [pixel]
    # Note: The sign convention used here is appropriate
    # for illumination propagating in the -z direction.
    # This means that a particle forming an image in the
    # focal plane (z = 0) is located at positive z.
    # Accounting for this by flipping the axial coordinate
    # is equivalent to using a mirrored (left-handed)
    # coordinate system.

    for idx in prange(length):
        kx = k * (coordinates[0, idx] - r_p[0])
        ky = k * (coordinates[1, idx] - r_p[1])
        kz = k * (coordinates[2, idx] - r_p[2])
        # 2. geometric factors
        kz *= -1.  # z convention
        krho = math.sqrt(kx**2 + ky**2)
        kr = math.sqrt(krho**2 + kz**2)

        theta = math.atan2(krho, kz)
        phi = math.atan2(ky, kx)
        sintheta = math.sin(theta)
        costheta = math.cos(theta)
        sinphi = math.sin(phi)
        cosphi = math.cos(phi)
        sinkr = math.sin(kr)
        coskr = math.cos(kr)

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
        if kz > 0:
            factor = 1.*1.j
        elif kz < 0:
            factor = -1.*1.j
        else:
            factor = 0.*1.j
        if not bohren:
            factor = -1.*factor

        xi_nm2 = coskr + factor * sinkr  # \xi_{-1}(kr)
        xi_nm1 = sinkr - factor * coskr  # \xi_0(kr)

        # 2. Angular functions (4.47), page 95
        pi_nm1 = 0.     # \pi_0(\cos\theta)
        pi_n = 1.                   # \pi_1(\cos\theta)

        # 3. Vector spherical harmonics: [r,theta,phi]
        mo1nr = 0.j
        mo1nt = 0.j
        mo1np = 0.j
        ne1nr = 0.j
        ne1nt = 0.j
        ne1np = 0.j

        # storage for scattered field
        esr = 0.j
        est = 0.j
        esp = 0.j

        # COMPUTE field by summing partial waves
        for n in range(1, norders):
            n = np.float64(n)
            # upward recurrences ...
            # 4. Legendre factor (4.47)
            # Method described by Wiscombe (1980)

            swisc = pi_n * costheta
            twisc = swisc - pi_nm1
            tau_n = pi_nm1 - n * twisc  # -\tau_n(\cos\theta)

            # ... Riccati-Bessel function, page 478
            xi_n = (2. * n - 1.) * \
                (xi_nm1 / kr) - xi_nm2  # \xi_n(kr)

            # ... Deirmendjian's derivative
            dn = (n * xi_n) / kr - xi_nm1

            # vector spherical harmonics (4.50)
            mo1nt = pi_n * xi_n     # ... divided by cosphi/kr
            mo1np = tau_n * xi_n    # ... divided by sinphi/kr

            # ... divided by cosphi sintheta/kr^2
            ne1nr = n * (n + 1.) * pi_n * xi_n
            ne1nt = tau_n * dn      # ... divided by cosphi/kr
            ne1np = pi_n * dn       # ... divided by sinphi/kr

            mod = n % 4
            if mod == 1:
                fac = 1.j
            elif mod == 2:
                fac = -1.+0.j
            elif mod == 3:
                fac = -0.-1.j
            else:
                fac = 1.+0.j

            # prefactor, page 93
            en = fac * (2. * n + 1.) / \
                n / (n + 1.)

            # the scattered field in spherical coordinates (4.45)
            esr += (1.j * en * ab[int(n), 0]) * ne1nr
            est += (1.j * en * ab[int(n), 0]) * ne1nt
            esp += (1.j * en * ab[int(n), 0]) * ne1np
            esr -= (en * ab[int(n), 1]) * mo1nr
            est -= (en * ab[int(n), 1]) * mo1nt
            esp -= (en * ab[int(n), 1]) * mo1np

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
        esr *= cosphi * sintheta * radialfactor**2
        est *= cosphi * radialfactor
        esp *= sinphi * radialfactor

        # By default, the scattered wave is returned in spherical
        # coordinates.  Project components onto Cartesian coordinates.
        # Assumes that the incident wave propagates along z and
        # is linearly polarized along x

        if cartesian:
            ecx = esr * sintheta * cosphi
            ecx += est * costheta * cosphi
            ecx -= esp * sinphi

            ecy = esr * sintheta * sinphi
            ecy += est * costheta * sinphi
            ecy += esp * cosphi
            ecz = (esr * costheta -
                   est * sintheta)
            result[0, idx] = ecx
            result[1, idx] = ecy
            result[2, idx] = ecz
        else:
            result[0, idx] = esr
            result[1, idx] = est
            result[2, idx] = esp
        result[:, idx] *= phase


@njit(parallel=True, fastmath=False, cache=True)
def fasthologram(field, alpha, n, hologram):
    for idx in prange(n):
        e = field[:, idx]
        e[0] += 1
        e *= alpha
        i = e * np.conj(e)
        hologram[idx] = np.real(np.sum(i))


@njit(parallel=True, fastmath=False, cache=True)
def fastresiduals(holo, data, noise):
    return (holo - data) / noise


@njit(parallel=True, fastmath=False, cache=True)
def fastchisqr(holo, data, noise):
    chisqr = 0.
    for idx in prange(holo.size):
        chisqr += ((holo[idx] - data[idx]) / noise) ** 2
    return chisqr
