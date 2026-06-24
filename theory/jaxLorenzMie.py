'''Lorenz-Mie hologram and Jacobian computed with JAX.

Provides a JIT-compiled forward pass and the analytical Jacobian for
all five particle parameters via forward-mode automatic differentiation
(:func:`jax.jacfwd`).  Falls back to the NumPy implementation when
JAX is not installed.
'''

from pylorenzmie.theory.LorenzMie import LorenzMie
from pylorenzmie.theory.Sphere import Sphere
from pylorenzmie.lib.lmtypes import Coefficients, Coordinates, Field
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    _jax_available = True
except ImportError:
    _jax_available = False


if _jax_available:

    def _downward_rec(z, nmax):
        '''Downward recurrence for the Riccati-Bessel D1 logarithmic derivative.

        Computes D1[n] for n = 0..nmax starting from D1[nmax] = 0
        (Bohren & Huffman Eq. 4.88 / Yang 2003 Eq. 16b).

        Parameters
        ----------
        z : complex JAX scalar
            Argument: x*m for the sphere interior, x for the medium.
        nmax : int
            Number of terms (static; determines scan length).

        Returns
        -------
        D1 : jnp.ndarray, shape (nmax+1,), complex128
        '''
        def step(D1_n, n):
            D1_nm1 = n / z - 1. / (D1_n + n / z)
            return D1_nm1, D1_nm1

        ns = jnp.arange(nmax, 0, -1, dtype=complex)
        _, D1_rev = jax.lax.scan(step, 0. + 0.j, ns)
        # D1_rev[k] = D1[nmax-1-k]; flip restores ascending order
        return jnp.concatenate([jnp.flip(D1_rev), jnp.array([0. + 0.j])])

    def _upward_rec(x, D1_med, nmax):
        '''Upward recurrences for ψ_n, ζ_n, and D3_n (Yang 2003 Eqs. 18–21).

        Parameters
        ----------
        x : complex JAX scalar
            Size parameter (passed as complex to enable AD through sin/exp).
        D1_med : jnp.ndarray, shape (nmax+1,), complex128
            Logarithmic derivative D1 evaluated at z = x (medium).
        nmax : int
            Number of terms.

        Returns
        -------
        Psi, zeta, D3 : jnp.ndarray, each shape (nmax+1,), complex128
        '''
        Psi_0 = jnp.sin(x)
        zeta_0 = -1.j * jnp.exp(1.j * x)
        Psizeta_0 = 0.5 * (1. - jnp.exp(2.j * x))
        D3_0 = 1.j + 0. * x   # trace x so AD flows through dtype/shape

        def step(carry, n_int):
            Psi_prev, zeta_prev, Psizeta_prev, D3_prev = carry
            n = jnp.asarray(n_int, dtype=complex)
            nox = n / x
            D1_nm1 = D1_med[n_int - 1]
            D1_n = D1_med[n_int]
            Psi_n = Psi_prev * (nox - D1_nm1)
            zeta_n = zeta_prev * (nox - D3_prev)
            Psizeta_n = Psizeta_prev * (nox - D1_nm1) * (nox - D3_prev)
            D3_n = D1_n + 1.j / Psizeta_n
            return (Psi_n, zeta_n, Psizeta_n, D3_n), (Psi_n, zeta_n, D3_n)

        ns = jnp.arange(1, nmax + 1)
        _, (Psi_rest, zeta_rest, D3_rest) = jax.lax.scan(
            step, (Psi_0, zeta_0, Psizeta_0, D3_0), ns)

        Psi = jnp.concatenate([Psi_0[None], Psi_rest])
        zeta = jnp.concatenate([zeta_0[None], zeta_rest])
        D3 = jnp.concatenate([D3_0[None], D3_rest])
        return Psi, zeta, D3

    def _jax_mie_ab(a_p, n_p, k_p, n_m, wavelength, nmax):
        '''JAX-differentiable Mie scattering coefficients (Wiscombe–Yang).

        nmax is a static Python int so jax.lax.scan compiles at a fixed
        length. All arithmetic is differentiable w.r.t. the scalar inputs.

        Parameters
        ----------
        a_p : JAX float scalar
            Particle radius, μm.
        n_p : JAX float scalar
            Particle refractive index.
        k_p : JAX float scalar
            Particle absorption coefficient.
        n_m : JAX float scalar
            Medium refractive index.
        wavelength : JAX float scalar
            Vacuum wavelength, μm.
        nmax : int
            Number of partial-wave terms (static).

        Returns
        -------
        ab : jnp.ndarray, shape (nmax+1, 2), complex128
        '''
        k = 2. * jnp.pi * jnp.real(n_m) / wavelength  # rad/μm in medium
        x = k * a_p                                      # size parameter
        m = (n_p + 1.j * k_p) / n_m                     # relative index

        D1_sphere = _downward_rec(x * m, nmax)           # Ha = Hb (single layer)
        D1_med = _downward_rec(x + 0.j, nmax)
        Psi, zeta, _ = _upward_rec(x + 0.j, D1_med, nmax)

        n = jnp.arange(nmax + 1, dtype=complex)
        Psir = jnp.roll(Psi, 1)
        zetar = jnp.roll(zeta, 1)

        fac_a = D1_sphere / m + n / x                    # Eq. 5
        ab_a = (fac_a * Psi - Psir) / (fac_a * zeta - zetar)
        fac_b = D1_sphere * m + n / x                    # Eq. 6
        ab_b = (fac_b * Psi - Psir) / (fac_b * zeta - zetar)

        ab = jnp.stack([ab_a, ab_b], axis=1)
        return ab.at[0].set(0. + 0.j)

    def _jax_lorenzmie(ab, kdr, bohren=True):
        '''JAX partial-wave sum for the scattered electric field.

        Implements Bohren & Huffman §4.4 (Eqs. 4.45–4.50) using
        jax.lax.scan over multipole orders.  Fully differentiable
        w.r.t. ab and kdr.

        Parameters
        ----------
        ab : jnp.ndarray, shape (norders, 2), complex128
            Mie scattering coefficients.
        kdr : jnp.ndarray, shape (3, npts), float64
            Wavenumber-scaled displacement from particle to coordinates.
        bohren : bool
            Selects h_n^(1) (True, default) or h_n^(2) (False).

        Returns
        -------
        field : jnp.ndarray, shape (3, npts), complex128
            Cartesian scattered field at each coordinate.
        '''
        kx = kdr[0]
        ky = kdr[1]
        kz = -kdr[2]   # flip: particle above focal plane is +z
        npts = kx.shape[0]

        krho = jnp.hypot(kx, ky)
        kr = jnp.hypot(krho, kz)
        # Compute cosφ/sinφ directly to avoid arctan2 singularity at kx=ky=0.
        # jnp.where selects the constant branch (gradient=0) at the on-axis
        # pixel, preventing 0/0 NaN in the analytical Jacobian.
        krho_safe = jnp.where(krho > 0., krho, 1.)
        cosphi = jnp.where(krho > 0., kx / krho_safe, 1.)
        sinphi = jnp.where(krho > 0., ky / krho_safe, 0.)
        theta = jnp.arctan2(krho, kz)
        costheta = jnp.cos(theta)
        sintheta = jnp.sin(theta)
        sinkr = jnp.sin(kr)
        coskr = jnp.cos(kr)

        sgn = 1.j * jnp.sign(kz) if bohren else -1.j * jnp.sign(kz)
        xi_nm2 = (coskr + sgn * sinkr).astype(complex)  # xi_{-1}(kr)
        xi_nm1 = (sinkr - sgn * coskr).astype(complex)  # xi_0(kr)

        norders = ab.shape[0]

        init = (
            jnp.zeros(npts, dtype=complex),    # pi_nm1
            jnp.ones(npts, dtype=complex),     # pi_n
            xi_nm2,
            xi_nm1,
            jnp.zeros((3, npts), dtype=complex),  # Es (spherical)
            jnp.ones((), dtype=complex),           # En_factor (tracks i^n)
        )

        def step(carry, n_int):
            pi_nm1, pi_n, xi_nm2, xi_nm1, Es, En_fac = carry
            n = jnp.asarray(n_int, dtype=complex)

            # Legendre upward recurrence (Wiscombe 1980)
            swisc = pi_n * costheta
            twisc = swisc - pi_nm1
            tau_n = pi_nm1 - n * twisc

            # Riccati-Bessel upward recurrence
            xi_n = (2. * n - 1.) * xi_nm1 / kr - xi_nm2
            Dn = n * xi_n / kr - xi_nm1

            # vector spherical harmonics (geometric factors divided out)
            Mo1n_t = pi_n * xi_n
            Mo1n_p = tau_n * xi_n
            Ne1n_r = (n * n + n) * Mo1n_t
            Ne1n_t = tau_n * Dn
            Ne1n_p = pi_n * Dn

            En_fac = En_fac * 1.j
            En = En_fac * (2. * n + 1.) / (n * n + n)
            an = 1.j * En * ab[n_int, 0]
            bn = En * ab[n_int, 1]

            new_Es = Es.at[0].add(an * Ne1n_r)
            new_Es = new_Es.at[1].add(an * Ne1n_t - bn * Mo1n_t)
            new_Es = new_Es.at[2].add(an * Ne1n_p - bn * Mo1n_p)

            new_pi_n = (swisc + (1. + 1. / n) * twisc).astype(complex)
            return (pi_n, new_pi_n, xi_nm1, xi_n, new_Es, En_fac), None

        (_, _, _, _, Es, _), _ = jax.lax.scan(
            step, init, jnp.arange(1, norders))

        # restore geometric factors
        Es = Es.at[0].multiply(cosphi * sintheta / (kr * kr))
        Es = Es.at[1].multiply(cosphi / kr)
        Es = Es.at[2].multiply(sinphi / kr)

        # project spherical → Cartesian
        return jnp.stack([
            Es[0] * sintheta * cosphi + Es[1] * costheta * cosphi - Es[2] * sinphi,
            Es[0] * sintheta * sinphi + Es[1] * costheta * sinphi + Es[2] * cosphi,
            Es[0] * costheta - Es[1] * sintheta,
        ])

    def _jax_hologram_core(r_p, a_p, n_p, k_p, k, n_m, wavelength,
                            coordinates, nmax):
        '''Pure JAX hologram, differentiable w.r.t. all non-static args.

        Parameters
        ----------
        r_p : jnp.ndarray, shape (3,)
            Particle position (pixels).
        a_p, n_p, k_p : JAX float scalars
            Sphere radius (μm), refractive index, absorption.
        k : JAX float scalar
            Wavenumber in rad/pixel.
        n_m : JAX float scalar
            Medium refractive index.
        wavelength : JAX float scalar
            Vacuum wavelength, μm.
        coordinates : jnp.ndarray, shape (3, npts)
            Pixel coordinates at which to evaluate the hologram.
        nmax : int
            Number of Mie terms (static).

        Returns
        -------
        hologram : jnp.ndarray, shape (npts,), float64
        '''
        ab = _jax_mie_ab(a_p, n_p, k_p, n_m, wavelength, nmax)
        kdr = k * (coordinates - r_p[:, None])
        field = _jax_lorenzmie(ab, kdr)
        field = field * jnp.exp(-1.j * k * r_p[2])
        field = field.at[0].add(1.)
        return jnp.sum(field.real**2 + field.imag**2, axis=0)

    # nmax (arg index 8) is static: controls lax.scan length at compile time
    _jax_hologram_jit = jax.jit(_jax_hologram_core, static_argnums=(8,))

    # Jacobian w.r.t. r_p (0), a_p (1), n_p (2); nmax (8) still static.
    # Module-level JIT avoids per-call retracing that a closure would cause.
    _jax_hologram_jac = jax.jit(
        jax.jacfwd(_jax_hologram_core, argnums=(0, 1, 2)),
        static_argnums=(8,),
    )


class jaxLorenzMie(LorenzMie):
    '''LorenzMie with JAX JIT compilation and analytical Jacobian.

    Overrides :meth:`hologram` and :meth:`lorenzmie` with JAX
    implementations compiled via :func:`jax.jit`.  Adds :meth:`jac`
    for the full analytical Jacobian via forward-mode automatic
    differentiation (:func:`jax.jacfwd`), covering all five particle
    parameters in five forward passes through the model.

    The class attribute ``method = 'jax numpy'`` allows
    :class:`~pylorenzmie.analysis.Optimizer` to select a compatible
    model.

    Parameters
    ----------
    coordinates, particle, instrument
        Forwarded to :class:`LorenzMie`.

    Notes
    -----
    Requires JAX with 64-bit support.  ``jax_enable_x64`` is set at
    import time, which affects the global JAX session.

    The first :meth:`hologram` (or :meth:`jac`) call for a given *nmax*
    triggers JIT compilation (typically < 1 s).  Calls with the same
    particle size reuse the compiled function via XLA's compilation cache.

    Only single :class:`~pylorenzmie.theory.Sphere` particles are
    JAX-accelerated.  Multi-particle configurations fall back to the
    NumPy implementation automatically.
    '''

    method: str = 'jax numpy'
    jac_params: frozenset = frozenset({'x_p', 'y_p', 'z_p', 'a_p', 'n_p'})

    def _compute_nmax(self) -> int:
        # Use the unscaled (rad/μm) wavenumber — same as Sphere.mie_coefficients
        k = float(self.instrument.wavenumber(scaled=False))
        x = k * float(self.particle.a_p)
        m = (self.particle.n_p + 1.j * self.particle.k_p) / self.instrument.n_m
        return Sphere.wiscombe_yang(x, m)

    def hologram(self, **kwargs) -> np.ndarray:
        '''JIT-compiled hologram for a single :class:`~pylorenzmie.theory.Sphere`.

        Falls back to the NumPy implementation for non-Sphere or
        multi-particle models, or when JAX is not installed.
        The ``cartesian`` and ``bohren`` keyword arguments are ignored
        when the JAX path is active (both are always True).
        '''
        if not _jax_available or not isinstance(self.particle, Sphere):
            return super().hologram(**kwargs)

        nmax = self._compute_nmax()
        r_p = jnp.asarray(self.particle.r_p + self.particle.r_0)
        a_p = jnp.asarray(float(self.particle.a_p))
        n_p = jnp.asarray(float(self.particle.n_p))
        k_p = jnp.asarray(float(self.particle.k_p))
        k = jnp.asarray(float(self.instrument.wavenumber()))
        n_m = jnp.asarray(float(self.instrument.n_m))
        wavelength = jnp.asarray(float(self.instrument.wavelength))
        coords = jnp.asarray(self.coordinates)

        return np.asarray(_jax_hologram_jit(
            r_p, a_p, n_p, k_p, k, n_m, wavelength, coords, nmax))

    def jac(self) -> dict:
        '''Analytical Jacobian of the hologram w.r.t. particle parameters.

        Uses :func:`jax.jacfwd` (5 forward-mode passes) to compute
        dH/d(x_p, y_p, z_p, a_p, n_p) at the current model state.
        *nmax* is fixed at its current value (from the Wiscombe–Yang
        criterion); the result is exact at the current operating point
        and valid for local optimization.

        Returns
        -------
        J : dict of str → numpy.ndarray, shape (npts,), float64
            Keys: ``'x_p'``, ``'y_p'``, ``'z_p'``, ``'a_p'``, ``'n_p'``.

        Raises
        ------
        RuntimeError
            If JAX is not installed.
        TypeError
            If :attr:`particle` is not a single :class:`~pylorenzmie.theory.Sphere`.
        '''
        if not _jax_available:
            raise RuntimeError('jax is required for jac()')
        if not isinstance(self.particle, Sphere):
            raise TypeError('jac() supports single Sphere particles only')

        nmax = self._compute_nmax()
        r_p = jnp.asarray(self.particle.r_p + self.particle.r_0)
        a_p = jnp.asarray(float(self.particle.a_p))
        n_p = jnp.asarray(float(self.particle.n_p))
        k_p = jnp.asarray(float(self.particle.k_p))
        k = jnp.asarray(float(self.instrument.wavenumber()))
        n_m = jnp.asarray(float(self.instrument.n_m))
        wavelength = jnp.asarray(float(self.instrument.wavelength))
        coords = jnp.asarray(self.coordinates)

        J_r, J_a, J_n = _jax_hologram_jac(
            r_p, a_p, n_p, k_p, k, n_m, wavelength, coords, nmax)

        return {
            'x_p': np.asarray(J_r[:, 0]),
            'y_p': np.asarray(J_r[:, 1]),
            'z_p': np.asarray(J_r[:, 2]),
            'a_p': np.asarray(J_a),
            'n_p': np.asarray(J_n),
        }


if __name__ == '__main__':  # pragma: no cover
    jaxLorenzMie.example()
