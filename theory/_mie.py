'''The recurrences of the Mie coefficients of a homogeneous sphere.

They are loops over the partial waves, each step a handful of complex
operations, so that the time of the pure-Python loop goes into the
interpreter.  They are compiled with Numba when it is installed, and run as
plain Python when it is not: the numbers are the same.

The equation numbers are those of Yang, *Appl. Opt.* **42**, 1710 (2003).
'''

import numpy as np

from pylorenzmie.lib.lmtypes import Coefficients

try:
    from numba import njit
except ImportError:                     # Numba is an optional extra
    def njit(*args, **kwargs):
        return lambda function: function


@njit(cache=True)
def start_margin(nmax: int) -> int:
    '''Orders above ``nmax`` where the downward recurrence starts.

    The logarithmic derivative ``D1`` is computed downward, from an order
    where it is set to zero.  The error of that start dies out only slowly
    on the way down, while the order is within the transition region of the
    Bessel functions, which is as wide as the cube root of the order.  A
    start at ``nmax`` leaves an error of 1e-4 in the first coefficients of a
    1.5 μm sphere of polystyrene in water, and 0.7% for 3 μm; Yang (2003)
    and Bohren and Huffman start 15 orders higher, which is not enough for
    a large sphere of a high index.  This margin, ``15 + 6 nmax^(1/3)``,
    leaves an error below 1e-13 for radii up to 50 μm and indices up to 3
    (``nmax`` is at least as large as the size parameter in the sphere).

    Parameters
    ----------
    nmax : int
        Number of partial waves to keep.

    Returns
    -------
    margin : int
        Number of orders to start above ``nmax``.
    '''
    return int(15 + 6. * np.cbrt(nmax))


@njit(cache=True, error_model='numpy')
def mie_ab(x: float, m: complex, nmax: int) -> Coefficients:
    '''Mie scattering coefficients of a sphere.

    Parameters
    ----------
    x : float
        Size parameter of the sphere, in the medium.
    m : complex
        Refractive index of the sphere relative to the medium.
    nmax : int
        Number of partial waves to keep; see
        :meth:`~pylorenzmie.theory.Sphere.Sphere.wiscombe_yang`.

    Returns
    -------
    ab : numpy.ndarray, shape (nmax + 1, 2), dtype complex
        The coefficients ``a_n`` and ``b_n``, with ``ab[0] = 0``.
    '''
    start = nmax + start_margin(nmax)
    ab = np.zeros((nmax + 1, 2), np.complex128)
    D1 = np.zeros(start + 1, np.complex128)
    D3 = np.zeros(nmax + 1, np.complex128)
    Ψ = np.zeros(nmax + 1, np.complex128)
    ζ = np.zeros(nmax + 1, np.complex128)

    # logarithmic derivative inside the sphere, downward from beyond the core
    z = x * m
    for n in range(start, 0, -1):
        D1[n-1] = n/z - 1./(D1[n] + n/z)             # Eq. (16b)
    Ha = D1[:nmax + 1].copy()                        # Eq. (7a)
    Hb = Ha                                          # Eq. (8a)

    # and in the medium
    D1[start] = 0.
    for n in range(start, 0, -1):
        D1[n-1] = n/x - 1./(D1[n] + n/x)             # Eq. (16b)

    # upward recurrence for Ψ, ζ, Ψζ and D3
    D3[0] = 1.j                                      # Eq. (18b)
    Ψ[0] = np.sin(x)                                 # Eq. (20a)
    ζ[0] = -1.j * np.exp(1.j * x)                    # Eq. (21a)
    Ψζ = 0.5 * (1. - np.exp(2.j * x))                # Eq. (18a)
    for n in range(1, nmax + 1):
        Ψ[n] = Ψ[n-1] * (n/x - D1[n-1])              # Eq. (20b)
        ζ[n] = ζ[n-1] * (n/x - D3[n-1])              # Eq. (21b)
        Ψζ *= (n/x - D1[n-1]) * (n/x - D3[n-1])      # Eq. (18c)
        D3[n] = D1[n] + 1.j/Ψζ                       # Eq. (18d)

    # scattering coefficients
    for n in range(1, nmax + 1):
        fac = Ha[n]/m + n/x                          # Eq. (5)
        ab[n, 0] = (fac*Ψ[n] - Ψ[n-1]) / (fac*ζ[n] - ζ[n-1])
        fac = Hb[n]*m + n/x                          # Eq. (6)
        ab[n, 1] = (fac*Ψ[n] - Ψ[n-1]) / (fac*ζ[n] - ζ[n-1])
    return ab
