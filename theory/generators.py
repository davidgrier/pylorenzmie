'''Generating functions of the primary Seidel aberrations.

Each aberration is the flow of a dimensionless vector field
:math:`\\vec{g}_n(\\vec{\\rho}) = \\nabla_{\\vec{p}} D_n` on the image
plane, where :math:`\\vec{\\rho}` is the in-plane displacement from the
scattering particle, :math:`R = \\sqrt{\\rho^2 + z_p^2}`, the
canonical momentum is :math:`\\vec{p} = n_m \\vec{\\rho} / R`, and
:math:`\\vec{\\eta} = (\\vec{r} - \\vec{r}_c) / \\lambda
= \\vec{q} + \\vec{\\rho} / \\lambda` is the position of the ray in the
image plane, measured from the aberration center in units of the vacuum
wavelength :math:`\\lambda`.  The ray coordinate :math:`\\vec{\\eta}`
differs from point to point of the image;
:math:`\\vec{q} = (\\vec{r}_p - \\vec{r}_c) / \\lambda` is its value at the
particle.
The generators :math:`D_n` are those of Table I of Gronert et al.
'''
from collections.abc import Callable, Sequence

import numpy as np

NAMES = ('defocus', 'spherical', 'coma', 'astigmatism',
         'fieldcurvature', 'distortion')


def generator(name: str,
              rho: np.ndarray,
              z_p: float,
              n_m: float,
              q: Sequence[float] = (0., 0.),
              scale: float = 1.) -> np.ndarray:
    '''Generating function g_n(rho).

    Parameters
    ----------
    name : str
        One of ``NAMES``.
    rho : numpy.ndarray, shape (2, npts)
        In-plane displacements from the particle [pixels].
    z_p : float
        Height of the particle above the focal plane [pixels].
    n_m : float
        Refractive index of the medium.
    q : array_like, shape (2,), optional
        Displacement of the particle from the aberration center,
        in vacuum wavelengths. Default: ``(0, 0)``.
        Used by the field-dependent aberrations only.
    scale : float, optional
        Vacuum wavelength [pixels]. The ray coordinate at ``rho`` is
        ``q + rho / scale``. Default: 1.
        Used by the field-dependent aberrations only.

    Returns
    -------
    g : numpy.ndarray, shape (2, npts)
        Dimensionless displacement per unit coefficient.
    '''
    R = np.sqrt(rho[0]**2 + rho[1]**2 + z_p**2)
    p = n_m * rho / R
    p2 = p[0]**2 + p[1]**2
    if name == 'defocus':                      # D = |p|^2
        return 2. * p
    if name == 'spherical':                    # D = |p|^4
        return 4. * p2 * p
    eta = np.asarray(q, dtype=float)[:, None] + rho / scale
    pe = p[0] * eta[0] + p[1] * eta[1]
    e2 = eta[0]**2 + eta[1]**2
    if name == 'coma':                         # D = |p|^2 (p.eta)
        return 2. * pe * p + p2 * eta
    if name == 'astigmatism':                  # D = (p.eta)^2
        return 2. * pe * eta
    if name == 'fieldcurvature':               # D = |p|^2 |eta|^2
        return 2. * e2 * p
    if name == 'distortion':                   # D = (p.eta) |eta|^2
        return e2 * eta
    raise ValueError(f'unknown aberration: {name!r}')


def flow(rho: np.ndarray,
         alpha: float,
         g: Callable[[np.ndarray], np.ndarray],
         nsteps: int = 4) -> np.ndarray:
    '''Fourth-order Runge-Kutta integration of d rho / d alpha = g(rho).

    Parameters
    ----------
    rho : numpy.ndarray, shape (2, npts)
        Starting points.
    alpha : float
        Final value of the flow parameter (same units as *rho*).
    g : callable
        Vector field, ``g(rho) -> (2, npts)``.
    nsteps : int, optional
        Number of Runge-Kutta steps. Default: 4.

    Returns
    -------
    numpy.ndarray, shape (2, npts)
        The points :math:`T_\\alpha(\\rho)`.
    '''
    h = alpha / nsteps
    x = rho
    for _ in range(nsteps):
        k1 = g(x)
        k2 = g(x + 0.5 * h * k1)
        k3 = g(x + 0.5 * h * k2)
        k4 = g(x + h * k3)
        x = x + h * (k1 + 2. * (k2 + k3) + k4) / 6.
    return x
