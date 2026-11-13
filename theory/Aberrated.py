'''Aberrations of the microscope as phase factors on the scattered field.

:func:`Aberrated` takes any calculator derived from
:class:`~pylorenzmie.theory.LorenzMie.LorenzMie` and returns a subclass that
multiplies the scattered field of each particle by a phase factor, leaving the
reference wave alone.  The factor is built from the five primary Seidel
aberrations: defocus, spherical aberration, coma, astigmatism and distortion.
Each coefficient is expressed in wavelengths of the illumination in vacuum
and enters through the phase factor of the operator formalism for the
aberration, which depends on the particle's height ``z_p`` and on the
distance from the particle in the image.  Coma, astigmatism and distortion
also depend on the position of each ray in the image plane relative to the
aberration center (``x_c``, ``y_c``, in pixels), in units of the vacuum
wavelength.

A sixth parameter, ``piston``, is the constant term of the phase.  It is the
phase of the scattered field relative to the reference wave, in wavelengths
of the illumination in vacuum like the other coefficients, and it multiplies
the scattered field by :math:`\\exp(2\\pi i\\,\\text{piston})`.  A constant
phase generates no ray deflection, so it has no generating function in the
operator formalism and no row in the table of aberrations, and it is not
affected by the refractive index of the medium; it is for instruments in
which the scattered and unscattered light do not reach the camera with the
phase that the ideal model assumes.  It is zero by default.

:data:`AberratedLorenzMie` is the aberrated NumPy calculator.  The class that
``pylorenzmie.theory`` exports under that name is built on the calculator that
``pylorenzmie.theory.LorenzMie`` selects (Numba if it is installed), so it is as
fast as the plain model.  The calculators that evaluate the field inside a
compiled kernel (Metal and JAX) apply the same phase inside the kernel.  Defocus is nearly degenerate with
``z_p``, so fit it only when the height is constrained by other means.
'''

from pylorenzmie.theory.LorenzMie import LorenzMie
from pylorenzmie.theory.Particle import Particle
from pylorenzmie.lib.lmtypes import Coordinates, Field, Properties
import numpy as np


def Aberrated(base_class: type) -> type:
    '''Return an aberrated subclass of a LorenzMie calculator.

    The returned class extends ``scattered_field`` to multiply
    the scattered field by a phase mask built from the primary Seidel
    aberrations (defocus, spherical, coma, astigmatism, distortion).

    Parameters
    ----------
    base_class : type
        A :class:`~pylorenzmie.theory.LorenzMie` subclass to extend.

    Returns
    -------
    AberratedLorenzMie : type
        New class inheriting from *base_class*.
    '''

    class AberratedLorenzMie(base_class):
        '''LorenzMie subclass with geometric aberrations.

        Inherits from the base class supplied to :func:`Aberrated`.

        Parameters
        ----------
        defocus : float, optional
            Defocus coefficient [wavelengths]. Default: 0.
        spherical : float, optional
            Spherical-aberration coefficient [wavelengths]. Default: 0.
        coma : float, optional
            Coma coefficient [wavelengths]. Default: 0.
        astigmatism : float, optional
            Astigmatism coefficient [wavelengths]. Default: 0.
        distortion : float, optional
            Distortion coefficient [wavelengths]. Default: 0.
        piston : float, optional
            Constant phase of the scattered field relative to the
            reference field [wavelengths]. Default: 0.
        x_c : float, optional
            x coordinate of the aberration center, in pixels.
            Used by coma, astigmatism, and distortion only.
            Default: 0.
        y_c : float, optional
            y coordinate of the aberration center, in pixels.
            Used by coma, astigmatism, and distortion only.
            Default: 0.

        Notes
        -----
        ``piston`` is not a geometric aberration.  It multiplies the
        scattered field, and not the reference field, by
        :math:`\\exp(2\\pi i\\,\\text{piston})`, which shifts the phase of
        the interference fringes by the same amount everywhere.  (A phase
        applied to both fields would have no effect on the hologram.)
        A piston of 1 is one vacuum wavelength of optical path, with no
        power of :math:`n_m`, consistent with the other coefficients, whose
        powers of :math:`n_m` grow with the order of the aberration.  The
        model of an ideal system fixes the relative phase through the
        propagation phase :math:`k z_p`; a real optical train need not.
        Treat a fitted piston as a diagnostic as well as a correction: it
        is nearly degenerate with the phase of the particle's
        forward-scattering amplitude, and so with its refractive index, and
        can absorb other model errors.  It is held fixed by default.

        Each coefficient is expressed in
        wavelengths of the illumination, :math:`\\lambda`
        (``instrument.wavelength``, the *vacuum* wavelength): a
        coefficient of 1 corresponds to an
        aberration amplitude :math:`\\alpha_n = \\lambda` in the
        operator formalism of [1]_, not to one wavelength of classical
        wavefront error at the edge of the pupil. The two conventions
        differ by an aberration-dependent power of :math:`n_m`, the
        medium's refractive index: for example, ``spherical = 1``
        corresponds to :math:`8\\pi n_m^4 \\approx 12.9` wavelengths
        (at :math:`n_m = 1.34`) of classical edge-of-pupil OPD, not 1.
        This is a deliberate choice: the coefficients of defocus and
        spherical aberration fitted here are the same numbers reported
        in [1]_, with no further conversion.  For simplicity, the position
        of a ray is measured in units of the vacuum wavelength here,
        and in units of the radius :math:`H` of the field of view in [1]_,
        so that the coefficients of coma, astigmatism and distortion
        differ by the factor :math:`(H/\\lambda)^k` given below.

        The phase contributed by each term, for a particle at pixel
        position :math:`\\vec{r}_p = (x_p, y_p, z_p)`, is

        .. math::

            \\Phi_\\text{defocus} &=
                4\\pi n_m^2\\, d\\, u \\\\
            \\Phi_\\text{spherical} &=
                8\\pi n_m^4\\, s\\, u^2 \\\\
            \\Phi_\\text{coma} &=
                6\\pi n_m^3\\, c\\, u^{3/2}\\,
                \\hat{\\vec{r}}\\cdot\\vec{\\eta} \\\\
            \\Phi_\\text{astigmatism} &=
                4\\pi n_m^2\\, a\\, u\\,
                (\\hat{\\vec{r}}\\cdot\\vec{\\eta})^2 \\\\
            \\Phi_\\text{distortion} &=
                2\\pi n_m\\, t\\, u^{1/2}\\, \\left|\\vec{\\eta}\\right|^2\\,
                \\hat{\\vec{r}}\\cdot\\vec{\\eta}

        where :math:`d, s, c, a, t` are ``defocus``, ``spherical``,
        ``coma``, ``astigmatism``, and ``distortion``,
        :math:`u = r^2/(r^2+z_p^2)` with :math:`r` the in-plane
        distance from the particle to the field point,
        :math:`\\hat{\\vec{r}}` is the unit vector from the particle to the
        field point, and
        :math:`\\vec{\\eta} = (\\vec{r} - \\vec{r}_c)/\\lambda` is the
        position of the ray at the field point :math:`\\vec{r}`, measured
        from the aberration center :math:`\\vec{r}_c = (x_c, y_c)` in
        units of the vacuum wavelength.  The ray coordinate
        :math:`\\vec{\\eta}` takes a different value at each point of the
        image, so that it is not the position of the particle: the
        aberrations of the optical train act on the field in the image
        plane, and a ray at that plane has the position and the momentum
        that these factors use.  The coefficients are
        instrumental constants, independent of the particle and of its
        position.
        Defocus and spherical aberration are rotationally symmetric
        about the optical axis and do not depend on :math:`(x_c, y_c)`.

        Table I of [1]_ gives the generating functions and phase
        factors in terms of the same quantities.

        The coefficients are related to the Seidel coefficients
        :math:`W_{k\\ell m}` of lens design, the optical path differences
        at the edge of the pupil and of the field, by

        .. math::

            W_{k\\ell m} / \\lambda =
                \\ell\\, \\text{NA}^\\ell\\, (H/\\lambda)^k\\, c_n

        where :math:`c_n` is the coefficient (in wavelengths),
        NA the numerical aperture of the objective,
        :math:`H` the radius of its field of view, and :math:`k` and
        :math:`\\ell` the degrees of the term in :math:`\\vec{\\eta}` and in
        the canonical momentum :math:`n_m \\sin\\beta`:
        :math:`(k, \\ell) = (0, 2), (0, 4), (1, 3), (2, 2), (3, 1)` for
        defocus, spherical aberration, coma, astigmatism and distortion.

        References
        ----------
        .. [1] M. K. Gronert, A. Prenowitz, K. Snyder, and D. G. Grier,
           "Operator-based aberration correction for holographic
           imaging" (2026).
        '''

        def __init__(self, *args,
                     defocus: float = 0.,
                     spherical: float = 0.,
                     coma: float = 0.,
                     astigmatism: float = 0.,
                     distortion: float = 0.,
                     piston: float = 0.,
                     x_c: float = 0.,
                     y_c: float = 0.,
                     **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.defocus = defocus
            self.spherical = spherical
            self.coma = coma
            self.astigmatism = astigmatism
            self.distortion = distortion
            self.piston = piston
            self.x_c = x_c
            self.y_c = y_c

        @LorenzMie.properties.getter
        def properties(self) -> Properties:
            return {**super().properties,
                    'defocus': self.defocus,
                    'spherical': self.spherical,
                    'coma': self.coma,
                    'astigmatism': self.astigmatism,
                    'distortion': self.distortion,
                    'piston': self.piston,
                    'x_c': self.x_c,
                    'y_c': self.y_c}

        def _aberration(self, r_p: Coordinates) -> Field:
            '''Aberration phase mask for a particle at r_p.'''
            if not (self.defocus or self.spherical or self.coma or
                    self.astigmatism or self.distortion or self.piston):
                return 1.

            n_m = self.instrument.n_m
            dx = self.coordinates[0] - r_p[0]
            dy = self.coordinates[1] - r_p[1]
            rsq = dx*dx + dy*dy
            zpsq = r_p[2]**2
            u = rsq / (rsq + zpsq)  # (r / sqrt(r^2 + z_p^2))^2

            phase = 0.
            if self.defocus:
                phase = phase + self.defocus * (4.*np.pi*n_m**2) * u
            if self.spherical:
                phase = phase + self.spherical * (8.*np.pi*n_m**4) * u**2

            if self.coma or self.astigmatism or self.distortion:
                # ray coordinate eta = (d + rho)/scale at each point, with
                # d the particle's displacement from the aberration center
                # and rho its displacement from the particle.  With
                # w = rho . (d + rho) = scale rho . eta, every term is
                # a polynomial in (dx, dy, dxc, dyc) over a power of R, so
                # nothing divides by r.
                inst = self.instrument
                scale = inst.wavelength / inst.magnification
                ex = (r_p[0] - self.x_c) + dx
                ey = (r_p[1] - self.y_c) + dy
                w = dx*ex + dy*ey
                R = np.sqrt(rsq + zpsq)
                if self.coma:
                    phase = phase + self.coma * (6.*np.pi*n_m**3) * \
                        rsq * w / (scale * R**3)
                if self.astigmatism:
                    phase = phase + self.astigmatism * \
                        (4.*np.pi*n_m**2) * w**2 / (scale * R)**2
                if self.distortion:
                    phase = phase + self.distortion * \
                        (2.*np.pi*n_m) * (ex*ex + ey*ey) * w / \
                        (scale**3 * R)

            mask = np.exp(1j * (phase + 2.*np.pi*self.piston))
            if hasattr(self, '_device'):
                import torch
                # single precision: Apple GPUs have no complex128
                return torch.as_tensor(mask, dtype=torch.complex64,
                                       device=self._device)
            return mask

        def _aberration_coefficients(self) -> np.ndarray | None:
            '''Coefficients for the JAX and Metal kernels, which compute
            the phase themselves instead of calling :meth:`_aberration`.
            None when all aberrations are zero.  The piston is not among
            them: the kernels receive it separately, as ``self.piston``.'''
            if not (self.defocus or self.spherical or
                    self.coma or self.astigmatism or self.distortion):
                return None
            return np.array([self.defocus, self.spherical, self.coma,
                             self.astigmatism, self.distortion,
                             self.x_c, self.y_c,
                             self.instrument.wavelength /
                             self.instrument.magnification], dtype=float)

        def scattered_field(self,
                            particle: Particle,
                            **kwargs) -> Field:
            '''Scattered field including geometric aberrations.'''
            field = super().scattered_field(particle, **kwargs)
            if getattr(self, '_aberrations_in_kernel', False):
                return field                  # the kernel has applied them
            r_p = particle.r_p + particle.r_0
            return field * self._aberration(r_p)

        # lets a back end that applies the phase in its kernel (Metal) keep
        # its fast path for aberrated models
        scattered_field._kernel_aberration = True

    return AberratedLorenzMie


AberratedLorenzMie = Aberrated(LorenzMie)
# Fix __qualname__ so pickle can locate the class at module level.
# Factory-defined classes get __qualname__ =
# 'Aberrated.<locals>.AberratedLorenzMie', which Python cannot
# resolve during unpickling.
AberratedLorenzMie.__qualname__ = 'AberratedLorenzMie'
AberratedLorenzMie.__name__ = 'AberratedLorenzMie'


if __name__ == '__main__':  # pragma: no cover
    AberratedLorenzMie.example(spherical=0.9)
