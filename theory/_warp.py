'''Fused Numba kernels for the geometric warp of :mod:`Distorted`.

The points are processed in blocks that stay in cache.  Each
Runge-Kutta stage is a simple loop over the block with the generating
function of one aberration inlined into it, so that Numba can vectorize
one loop per aberration.  The flow of a single aberration is the "summed"
flow of that generator alone, so composed flows chain the same block
routine.
The kernels reproduce :func:`pylorenzmie.theory.generators.flow`
with the generators of :func:`pylorenzmie.theory.generators.generator`
(the integer *kinds* are indices into
``pylorenzmie.theory.generators.NAMES``).

Numba compiles a kernel the first time that it is called, which takes
seconds, and compiling needs a deep stack that the secondary threads of a
GUI toolkit do not have.  The kernels are therefore cached on disk, and
:func:`warm_up` compiles (or loads) all of them in the thread that calls it,
before any other thread needs one.
'''
import importlib
import threading

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

_mie = importlib.import_module('pylorenzmie.theory.numbaLorenzMie')

BLOCK = 2048          # points per block, sized to stay in cache

Vector = NDArray[np.float64]        # one coordinate of many points
Kinds = NDArray[np.int64]           # indices into generators.NAMES
Pair = tuple[float, float]          # (x, y) of one point or vector


def parallel_allowed(model: object, npts: int) -> bool:
    '''True if the warp may use all Numba threads.

    The rules are those of the Mie kernel of
    :class:`~pylorenzmie.theory.numbaLorenzMie.numbaLorenzMie`: the model
    asks for it with its ``parallel`` attribute, there are enough points,
    and the process was not created by ``fork``.
    '''
    return bool(getattr(model, 'parallel', False) and not _mie._FORKED and
                npts >= getattr(model, 'parallel_threshold', 2048))


def PARALLEL_LOCK() -> threading.Lock:
    '''The lock that serializes parallel Numba launches (shared with the
    Mie kernel, because the threading layer is shared).'''
    return _mie._PARALLEL_LOCK


# ------------------------------------------------------------------ generators
# Each generator g_n(x, y) is the vector field whose flow is one aberration.
# (x, y) is the in-plane displacement from the particle [pixels], z2 = z_p**2,
# n_m is the refractive index of the medium, and (q0, q1) is the displacement
# of the particle from the aberration center in units of z_p.  The function
# returns the (x, y) components of g_n; see generators.generator.
@njit(inline='always', fastmath=True, cache=True)
def _p(x: float, y: float, z2: float, n_m: float) -> Pair:
    inv = n_m / np.sqrt(x * x + y * y + z2)
    return x * inv, y * inv


@njit(inline='always', fastmath=True, cache=True)
def _g_defocus(x: float, y: float, z2: float, n_m: float,
               q0: float, q1: float) -> Pair:
    px, py = _p(x, y, z2, n_m)
    return 2. * px, 2. * py


@njit(inline='always', fastmath=True, cache=True)
def _g_spherical(x: float, y: float, z2: float, n_m: float,
                 q0: float, q1: float) -> Pair:
    px, py = _p(x, y, z2, n_m)
    p2 = px * px + py * py
    return 4. * p2 * px, 4. * p2 * py


@njit(inline='always', fastmath=True, cache=True)
def _g_coma(x: float, y: float, z2: float, n_m: float,
            q0: float, q1: float) -> Pair:
    px, py = _p(x, y, z2, n_m)
    p2 = px * px + py * py
    pq = px * q0 + py * q1
    return 2. * pq * px + p2 * q0, 2. * pq * py + p2 * q1


@njit(inline='always', fastmath=True, cache=True)
def _g_astigmatism(x: float, y: float, z2: float, n_m: float,
                   q0: float, q1: float) -> Pair:
    px, py = _p(x, y, z2, n_m)
    pq = px * q0 + py * q1
    return 2. * pq * q0, 2. * pq * q1


@njit(inline='always', fastmath=True, cache=True)
def _g_fieldcurvature(x: float, y: float, z2: float, n_m: float,
                      q0: float, q1: float) -> Pair:
    px, py = _p(x, y, z2, n_m)
    q2 = q0 * q0 + q1 * q1
    return 2. * q2 * px, 2. * q2 * py


@njit(inline='always', fastmath=True, cache=True)
def _g_distortion(x: float, y: float, z2: float, n_m: float,
                  q0: float, q1: float) -> Pair:
    q2 = q0 * q0 + q1 * q1
    return q2 * q0 + 0. * x, q2 * q1 + 0. * y


# --------------------------------------------------------- stage-wise flow
def _accumulator(generator):
    '''A kernel that adds alpha * g(x, y) to (kx, ky) for n points.

    The generator is part of the kernel, so Numba compiles (and can
    vectorize) one loop for each aberration, and can cache it.
    '''
    @njit(fastmath=True, cache=True)
    def accumulate(alpha: float,
                   x: Vector, y: Vector, kx: Vector, ky: Vector,
                   n: int, z2: float, n_m: float,
                   q0: float, q1: float) -> None:
        for i in range(n):
            gx, gy = generator(x[i], y[i], z2, n_m, q0, q1)
            kx[i] += alpha * gx
            ky[i] += alpha * gy

    return accumulate


_accumulate_defocus = _accumulator(_g_defocus)
_accumulate_spherical = _accumulator(_g_spherical)
_accumulate_coma = _accumulator(_g_coma)
_accumulate_astigmatism = _accumulator(_g_astigmatism)
_accumulate_fieldcurvature = _accumulator(_g_fieldcurvature)
_accumulate_distortion = _accumulator(_g_distortion)


@njit(fastmath=True, cache=True)
def _stage(kinds: Kinds, alphas: Vector,
           x: Vector, y: Vector, kx: Vector, ky: Vector,
           n: int, z2: float, n_m: float,
           q0: float, q1: float) -> None:
    '''kx, ky = sum_n alpha_n g_n(x, y) for the first n points.'''
    kx[:n] = 0.
    ky[:n] = 0.
    for j in range(kinds.size):
        kind = kinds[j]
        a = alphas[j]
        if kind == 0:
            _accumulate_defocus(a, x, y, kx, ky, n, z2, n_m, q0, q1)
        elif kind == 1:
            _accumulate_spherical(a, x, y, kx, ky, n, z2, n_m, q0, q1)
        elif kind == 2:
            _accumulate_coma(a, x, y, kx, ky, n, z2, n_m, q0, q1)
        elif kind == 3:
            _accumulate_astigmatism(a, x, y, kx, ky, n, z2, n_m, q0, q1)
        elif kind == 4:
            _accumulate_fieldcurvature(a, x, y, kx, ky, n, z2, n_m, q0, q1)
        else:
            _accumulate_distortion(a, x, y, kx, ky, n, z2, n_m, q0, q1)


@njit(fastmath=True, cache=True)
def _summed_block(kinds: Kinds, alphas: Vector,
                  x: Vector, y: Vector, lo: int, hi: int,
                  nsteps: int, z2: float, n_m: float,
                  q0: float, q1: float) -> tuple[Vector, Vector]:
    '''Runge-Kutta flow of points lo..hi through the summed generator.'''
    n = hi - lo
    xs = x[lo:hi].copy()
    ys = y[lo:hi].copy()
    h = 1. / nsteps
    k1x = np.empty(n)
    k1y = np.empty(n)
    k2x = np.empty(n)
    k2y = np.empty(n)
    k3x = np.empty(n)
    k3y = np.empty(n)
    k4x = np.empty(n)
    k4y = np.empty(n)
    tx = np.empty(n)
    ty = np.empty(n)
    for _ in range(nsteps):
        _stage(kinds, alphas, xs, ys, k1x, k1y, n, z2, n_m, q0, q1)
        for i in range(n):
            tx[i] = xs[i] + 0.5 * h * k1x[i]
            ty[i] = ys[i] + 0.5 * h * k1y[i]
        _stage(kinds, alphas, tx, ty, k2x, k2y, n, z2, n_m, q0, q1)
        for i in range(n):
            tx[i] = xs[i] + 0.5 * h * k2x[i]
            ty[i] = ys[i] + 0.5 * h * k2y[i]
        _stage(kinds, alphas, tx, ty, k3x, k3y, n, z2, n_m, q0, q1)
        for i in range(n):
            tx[i] = xs[i] + h * k3x[i]
            ty[i] = ys[i] + h * k3y[i]
        _stage(kinds, alphas, tx, ty, k4x, k4y, n, z2, n_m, q0, q1)
        for i in range(n):
            xs[i] += h * (k1x[i] + 2. * (k2x[i] + k3x[i]) + k4x[i]) / 6.
            ys[i] += h * (k1y[i] + 2. * (k2y[i] + k3y[i]) + k4y[i]) / 6.
    return xs, ys


# ------------------------------------------------------------------ drivers
def _make(parallel: bool) -> tuple:
    loop = prange if parallel else range

    @njit(fastmath=True, parallel=parallel, cache=True)
    def composed(x: Vector, y: Vector, kinds: Kinds, alphas: Vector,
                 nsteps: int, z_p: float, n_m: float,
                 q0: float, q1: float) -> tuple[Vector, Vector]:
        '''Points through the flows, the last listed operator first.

        The flow of one generator with coefficient alpha is the summed
        flow of that generator alone, so each aberration reuses the
        stage-wise block algorithm.
        '''
        n = x.size
        ox = np.empty(n)
        oy = np.empty(n)
        z2 = z_p * z_p
        nblocks = (n + BLOCK - 1) // BLOCK
        for b in loop(nblocks):
            lo = b * BLOCK
            hi = min(lo + BLOCK, n)
            xs = x[lo:hi].copy()
            ys = y[lo:hi].copy()
            for j in range(kinds.size - 1, -1, -1):
                xs, ys = _summed_block(kinds[j:j + 1], alphas[j:j + 1],
                                       xs, ys, 0, hi - lo, nsteps,
                                       z2, n_m, q0, q1)
            ox[lo:hi] = xs
            oy[lo:hi] = ys
        return ox, oy

    @njit(fastmath=True, parallel=parallel, cache=True)
    def summed(x: Vector, y: Vector, kinds: Kinds, alphas: Vector,
               nsteps: int, z_p: float, n_m: float,
               q0: float, q1: float) -> tuple[Vector, Vector]:
        '''Points through one flow of the summed generator.'''
        n = x.size
        ox = np.empty(n)
        oy = np.empty(n)
        z2 = z_p * z_p
        nblocks = (n + BLOCK - 1) // BLOCK
        for b in loop(nblocks):
            lo = b * BLOCK
            hi = min(lo + BLOCK, n)
            xs, ys = _summed_block(kinds, alphas, x, y, lo, hi, nsteps,
                                   z2, n_m, q0, q1)
            ox[lo:hi] = xs
            oy[lo:hi] = ys
        return ox, oy

    return composed, summed


warp_composed, warp_summed = _make(False)
warp_composed_parallel, warp_summed_parallel = _make(True)

_warmed = False


def warm_up() -> None:
    '''Compile the kernels, or load them from the cache.

    The first call in a process takes about a second with a warm cache and
    several seconds without one; later calls return at once.  Call it from
    the thread that sets up the model, so that no other thread compiles a
    kernel while it is running.  The parallel kernels are left out in a
    process created by ``fork``, which cannot use them.
    '''
    global _warmed
    if _warmed:
        return
    x = np.zeros(2)
    args = (x, x, np.zeros(1, dtype=np.int64), np.zeros(1), 1,
            1., 1., 0., 0.)
    warp_composed(*args)
    warp_summed(*args)
    if not _mie._FORKED:
        with PARALLEL_LOCK():
            warp_composed_parallel(*args)
            warp_summed_parallel(*args)
    _warmed = True
