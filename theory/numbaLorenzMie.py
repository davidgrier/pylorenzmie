from pylorenzmie.lib.lmtypes import Coefficients, Coordinates, Field
from pylorenzmie.theory.LorenzMie import LorenzMie
import numpy as np
import threading

from numba import njit, prange


def _partial_wave_constants(ab):
    '''Coefficients of the partial-wave sum that do not depend on the point.

    The Mie coefficients are folded together with the prefactors
    ``i**n (2n + 1) / (n (n + 1))`` of Bohren and Huffman (4.45), so that
    the loop over points needs no divisions and no complex prefactor
    recurrences.
    '''
    norders = ab.shape[0]
    n = np.arange(norders, dtype=np.float64)
    nn = np.maximum(n, 1.)
    En = (1.j ** np.arange(norders)) * (2. * n + 1.) / (nn * nn + nn)
    an = 1.j * En * ab[:, 0]
    bn = En * ab[:, 1]
    return (np.ascontiguousarray(an * (n * n + n)),     # multiplies Ne1n[0]
            np.ascontiguousarray(an),
            np.ascontiguousarray(bn),
            np.ascontiguousarray(2. * n - 1.),
            np.ascontiguousarray(1. + 1. / nn))


def _make_kernel(parallel: bool):
    loop = prange if parallel else range

    # error_model='numpy': a division by zero gives inf or NaN, as in the
    # NumPy implementation, instead of raising ZeroDivisionError
    @njit(fastmath=True, cache=True, parallel=parallel,
          error_model='numpy')
    def kernel(kdr, an0, an, bn, c2, c3, out, cartesian, bohren):
        '''Scattered field, one point at a time.

        The partial-wave sums of all points are independent, so each
        point is carried through the Riccati-Bessel and Legendre upward
        recurrences with scalar variables.  Nothing is allocated, and
        the loop over points can run in parallel.
        '''
        npts = kdr.shape[1]
        norders = an.shape[0]
        for i in loop(npts):
            # GEOMETRY
            # Illumination propagates along -z, so a particle above the
            # focal plane (z > 0) produces a diverging wave: flip kz.
            kx = kdr[0, i]
            ky = kdr[1, i]
            kz = -kdr[2, i]
            kρ = np.sqrt(kx * kx + ky * ky)
            kr = np.sqrt(kρ * kρ + kz * kz)
            inv_kr = 1. / kr
            sinkr = np.sin(kr)
            coskr = np.cos(kr)
            cosθ = kz * inv_kr
            sinθ = kρ * inv_kr
            if kρ > 0.:
                cosφ = kx / kρ
                sinφ = ky / kρ
            else:
                cosφ = 1.
                sinφ = 0.

            # Riccati-Bessel starting values, page 478.  The sign of the
            # imaginary part selects h_n^(1) (z > 0) or h_n^(2) (z < 0).
            if kz > 0.:
                s = 1.
            elif kz < 0.:
                s = -1.
            else:
                s = 0.
            if not bohren:
                s = -s
            factor = 1.j * s
            ξ_nm2 = coskr + factor * sinkr   # \xi_{-1}(kr)
            ξ_nm1 = sinkr - factor * coskr   # \xi_0(kr)

            # Angular function starting values (4.47), page 95
            π_nm1 = 0.                       # \pi_0(\cos\theta)
            π_n = 1.                         # \pi_1(\cos\theta)

            # scattered field in spherical coordinates (r, θ, φ)
            Er = 0.j
            Eθ = 0.j
            Eφ = 0.j
            for n in range(1, norders):
                # Legendre upward recurrence, Wiscombe (1980)
                swisc = π_n * cosθ
                twisc = swisc - π_nm1
                τ_n = π_nm1 - n * twisc           # -\tau_n(\cos\theta)

                # Riccati-Bessel upward recurrence, page 478
                ξ_n = c2[n] * (ξ_nm1 * inv_kr) - ξ_nm2
                Dn = n * (ξ_n * inv_kr) - ξ_nm1   # Deirmendjian

                # vector spherical harmonics (4.50)
                Mθ = π_n * ξ_n       # Mo1n[1], divided by cosφ/kr
                Mφ = τ_n * ξ_n       # Mo1n[2], divided by sinφ/kr
                Nθ = τ_n * Dn        # Ne1n[1], divided by cosφ/kr
                Nφ = π_n * Dn        # Ne1n[2], divided by sinφ/kr
                # Ne1n[0] = (n*n + n) * Mθ, folded into an0

                # the scattered field in spherical coordinates (4.45)
                Er += an0[n] * Mθ
                Eθ += an[n] * Nθ - bn[n] * Mθ
                Eφ += an[n] * Nφ - bn[n] * Mφ

                # upward recurrences
                π_nm1 = π_n
                π_n = swisc + c3[n] * twisc
                ξ_nm2 = ξ_nm1
                ξ_nm1 = ξ_n

            # restore geometric factors divided out of VSH for accuracy
            Er *= cosφ * sinθ * inv_kr * inv_kr
            Eθ *= cosφ * inv_kr
            Eφ *= sinφ * inv_kr

            if cartesian:       # incident wave along z, polarized along x
                out[0, i] = (Er * sinθ * cosφ + Eθ * cosθ * cosφ
                             - Eφ * sinφ)
                out[1, i] = (Er * sinθ * sinφ + Eθ * cosθ * sinφ
                             + Eφ * cosφ)
                out[2, i] = Er * cosθ - Eθ * sinθ
            else:
                out[0, i] = Er
                out[1, i] = Eθ
                out[2, i] = Eφ

    return kernel


_serial_kernel = _make_kernel(False)
_parallel_kernel = _make_kernel(True)

# Numba's default threading layer ('workqueue') aborts the process if a
# parallel kernel is entered by two Python threads at once.  Serializing the
# launches makes concurrent use safe with every threading layer.
_PARALLEL_LOCK = threading.Lock()


def compute_field_jit(kdr, buffers, scratch, ab, cartesian, bohren):
    '''Scattered field summed over Mie partial waves (compatibility wrapper).

    Parameters
    ----------
    kdr : numpy.ndarray, shape (3, npts)
        Wave-number-scaled displacement from particle to each
        coordinate point.
    buffers : sequence of numpy.ndarray
        Working arrays, each of shape (3, npts) and dtype complex.
        The field is written to ``buffers[3]`` (Cartesian) or
        ``buffers[2]`` (spherical).
    scratch : numpy.ndarray
        Unused. Kept for compatibility.
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
    out = buffers[3] if cartesian else buffers[2]
    _serial_kernel(np.ascontiguousarray(kdr, dtype=np.float64),
                   *_partial_wave_constants(ab), out, cartesian, bohren)
    return out


class numbaLorenzMie(LorenzMie):
    '''LorenzMie accelerated with Numba JIT compilation.

    Overrides :meth:`lorenzmie` with a Numba-compiled kernel that carries
    each point through the partial-wave recurrences with scalar
    variables, so that no temporary arrays are created.  (A kernel
    written with array expressions, as in the NumPy implementation,
    allocates and streams a temporary for every statement and is no
    faster than NumPy.)  The first call triggers JIT compilation
    (warm-up); subsequent calls use the cached compiled function.

    The class attribute ``method = 'numba numpy'`` allows :class:`Optimizer`
    to select a compatible model.

    Attributes
    ----------
    parallel : bool
        Evaluate on all Numba threads. Default: ``True``.
    parallel_threshold : int
        Smallest number of points evaluated in parallel; smaller
        evaluations run on one thread, where the cost of starting the
        threads would outweigh the gain. Default: 2048.

    Notes
    -----
    **Parallel execution.**
    The points of a hologram are independent of each other, so the
    kernel is parallel by default: on a ten-core machine it is about 3
    to 5 times faster than one thread of the same kernel for 10^4 or more
    points, which is roughly 4 times faster than the NumPy calculator on
    one thread and 10 to 20 times faster with threads. Parallel
    execution is the right choice when this
    calculator is the only thing using the processor, as in fitting
    one hologram after another. Fall back to one thread, with
    ``model.parallel = False`` for one model or ``numbaLorenzMie.parallel
    = False`` for all, when:

    * your code already runs many calculations at once, for example in
      a pool of worker processes (``multiprocessing``, ``joblib``,
      ``concurrent.futures``) with a worker per core. Threads in every
      worker then compete for the same cores, which is slower than one
      thread per worker. Either set ``parallel = False`` in the workers
      or share the cores by limiting the threads of each worker with
      :func:`numba.set_num_threads` or, before Numba is imported,
      the ``NUMBA_NUM_THREADS`` environment variable;
    * the calculator is called from inside another Numba parallel
      function (a ``prange`` loop). Nested parallel regions are not
      supported by Numba's default threading layer and abort the
      process;
    * you need the processor for something else at the same time, such
      as a responsive user interface, or reproducible timings.

    Calls from several Python threads at the same time are safe: they
    are serialized, because Numba's default ``workqueue`` threading
    layer aborts the process when two threads enter a parallel region at
    once. The calls then take turns using all of the cores. For
    simultaneous calls, use ``parallel = False`` in each thread.

    The field is undefined at the center of the particle, where
    :math:`kr = 0`. As in the NumPy implementation, such a point (a pixel
    that coincides with a particle at :math:`z_p = 0`) gives NaN, which
    propagates to the hologram, and no exception is raised.

    The field returned by :meth:`lorenzmie` is a working array that is
    overwritten by the next call.
    '''

    method: str = 'numba numpy'
    parallel: bool = True
    parallel_threshold: int = 2048

    def _allocate(self) -> None:
        super()._allocate()
        # only the two output buffers are used; free the others
        empty = np.empty((3, 0), dtype=complex)
        self.buffers[0] = empty
        self.buffers[1] = empty

    def lorenzmie(self,
                  ab: Coefficients,
                  kdr: Coordinates,
                  cartesian: bool = True,
                  bohren: bool = True) -> Field:
        kdr = np.ascontiguousarray(kdr, dtype=np.float64)
        out = self.buffers[3] if cartesian else self.buffers[2]
        if out.shape[1] != kdr.shape[1]:
            out = np.empty((3, kdr.shape[1]), dtype=complex)
        constants = _partial_wave_constants(ab)
        if self.parallel and kdr.shape[1] >= self.parallel_threshold:
            with _PARALLEL_LOCK:
                _parallel_kernel(kdr, *constants, out, cartesian, bohren)
        else:
            _serial_kernel(kdr, *constants, out, cartesian, bohren)
        return out


if __name__ == '__main__':  # pragma: no cover
    numbaLorenzMie.example()
