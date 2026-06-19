from pylorenzmie.lib.types import Coefficients, Coordinates, Field
from pylorenzmie.theory.LorenzMie import LorenzMie
import numpy as np

try:
    from numba import njit
except ImportError:
    from pylorenzmie.utilities.numba import njit


@njit(fastmath=True, cache=True)
def compute_field_jit(kdr, buffers, scratch, ab, cartesian, bohren):
    '''Numba-compiled scattered field sum over Mie partial waves.

    Parameters
    ----------
    kdr : numpy.ndarray, shape (3, npts)
        Wave-number-scaled displacement from particle to each
        coordinate point.
    buffers : tuple of numpy.ndarray, each shape (3, npts), dtype complex
        Pre-allocated working arrays (Mo1n, Ne1n, Es, Ec).
    scratch : numpy.ndarray, shape (2, npts), dtype float
        Pre-allocated real scratch arrays (swisc, twisc) for the
        Legendre upward recurrence.
    ab : numpy.ndarray, shape (n_orders, 2), dtype complex
        Mie scattering coefficients.
    cartesian : bool
        If True, return field projected onto Cartesian coordinates.
    bohren : bool
        If True, use sign convention from Bohren and Huffman.

    Returns
    -------
    field : numpy.ndarray, shape (3, npts), dtype complex
        Complex scattered field at each coordinate.
    '''
    norders = ab.shape[0]
    Mo1n, Ne1n, Es, Ec = buffers
    swisc = scratch[0]
    twisc = scratch[1]

    # GEOMETRY
    # Sign convention: illumination propagates along -z, so a particle
    # above the focal plane (z > 0) produces a diverging wave.
    # Flipping kz is equivalent to working in a mirrored coordinate system.
    kx = kdr[0, :]
    ky = kdr[1, :]
    kz = -kdr[2, :]
    npts = kx.shape[0]

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
    # Riccati-Bessel starting values, page 478.
    # Sign of imaginary part selects h_n^(1) (z>0) or h_n^(2) (z<0).
    if bohren:
        sign = 1.j * np.sign(kz)
    else:
        sign = -1.j * np.sign(kz)
    ξ_nm2 = coskr + sign * sinkr   # ξ_{-1}(kr)
    ξ_nm1 = sinkr - sign * coskr   # ξ_0(kr)

    # Angular function starting values (4.47), page 95
    π_nm1 = np.zeros(npts)   # π_0(cosθ)
    π_n = np.ones(npts)       # π_1(cosθ)

    Es.fill(0.j)

    En_factor = 1. + 0.j   # tracks 1.j**n iteratively

    # COMPUTE field by summing partial waves
    for n in range(1, norders):
        # Legendre upward recurrence — Wiscombe (1980)
        swisc[:] = π_n * cosθ
        twisc[:] = swisc - π_nm1
        τ_n = π_nm1 - n * twisc      # -τ_n(cosθ)

        # Riccati-Bessel upward recurrence, page 478
        ξ_n = (2.*n - 1.) * (ξ_nm1 / kr) - ξ_nm2

        # Deirmendjian's derivative
        Dn = n * (ξ_n / kr) - ξ_nm1

        # vector spherical harmonics (4.50)
        Mo1n[1] = π_n * ξ_n      # divided by cosφ/kr
        Mo1n[2] = τ_n * ξ_n      # divided by sinφ/kr
        Ne1n[0] = (n*n + n) * Mo1n[1]   # do not recompute π_n*ξ_n
        Ne1n[1] = τ_n * Dn       # divided by cosφ/kr
        Ne1n[2] = π_n * Dn       # divided by sinφ/kr

        # prefactor, page 93
        En_factor *= 1.j
        En = En_factor * (2.*n + 1.) / (n*n + n)
        an = 1.j * En * ab[n, 0]
        bn = En * ab[n, 1]

        # scattered field in spherical coordinates (4.45)
        # Mo1n[0] == 0 always; Es[0] has no Mo1n contribution
        Es[0] += an * Ne1n[0]
        Es[1] += an * Ne1n[1] - bn * Mo1n[1]
        Es[2] += an * Ne1n[2] - bn * Mo1n[2]

        # upward recurrences
        π_nm1 = π_n
        π_n = swisc + (1. + 1./n) * twisc
        ξ_nm2 = ξ_nm1
        ξ_nm1 = ξ_n

    # restore geometric factors divided out of VSH for accuracy
    Es[0] *= cosφ * sinθ / (kr * kr)
    Es[1] *= cosφ / kr
    Es[2] *= sinφ / kr

    # project to Cartesian (incident wave along z, polarized along x)
    if cartesian:
        Ec[0] = Es[0] * sinθ * cosφ + Es[1] * cosθ * cosφ - Es[2] * sinφ
        Ec[1] = Es[0] * sinθ * sinφ + Es[1] * cosθ * sinφ + Es[2] * cosφ
        Ec[2] = Es[0] * cosθ - Es[1] * sinθ
        return Ec
    else:
        return Es


class numbaLorenzMie(LorenzMie):
    '''LorenzMie accelerated with Numba JIT compilation.

    Overrides :meth:`lorenzmie` with a Numba-compiled kernel that runs
    the partial-wave sum in native code.  The first call triggers JIT
    compilation (warm-up); subsequent calls use the cached compiled
    function.

    The class attribute ``method = 'numba'`` allows :class:`Optimizer`
    to select a compatible model.
    '''

    method: str = 'numba'

    def _allocate(self) -> None:
        super()._allocate()
        npts = self.coordinates.shape[1]
        self._scratch = np.empty((2, npts))

    def lorenzmie(self,
                  ab: Coefficients,
                  kdr: Coordinates,
                  cartesian: bool = True,
                  bohren: bool = True) -> Field:
        return compute_field_jit(kdr, tuple(self.buffers), self._scratch,
                                 ab, cartesian, bohren)


if __name__ == '__main__':  # pragma: no cover
    numbaLorenzMie.example()
