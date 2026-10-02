from pylorenzmie.theory.LorenzMie import LorenzMie
from pylorenzmie.theory.Particle import Particle
from pylorenzmie.lib.lmtypes import Coordinates, Field, Properties
import numpy as np


def Aberrated(base_class: type) -> type:
    '''Return an aberrated subclass of a LorenzMie calculator.

    The returned class extends :meth:`scattered_field` to multiply
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
        Each coefficient is expressed in wavelengths of the
        illumination, :math:`\\lambda` (``instrument.wavelength``, the
        *vacuum* wavelength): a coefficient of 1 corresponds to an
        aberration amplitude :math:`\\alpha_n = \\lambda` in the
        operator formalism of [1]_, not to one wavelength of classical
        wavefront error at the edge of the pupil. The two conventions
        differ by an aberration-dependent power of :math:`n_m`, the
        medium's refractive index: for example, ``spherical = 1``
        corresponds to :math:`8\\pi n_m^4 \\approx 12.9` wavelengths
        (at :math:`n_m = 1.34`) of classical edge-of-pupil OPD, not 1.
        This is a deliberate choice: coefficients fitted here are the
        same numbers reported in [1]_, with no further conversion.

        The phase contributed by each term, for a particle at pixel
        position :math:`\\vec{r}_p = (x_p, y_p, z_p)`, is

        .. math::

            \\Phi_\\text{defocus} &=
                4\\pi n_m^2\\, d\\, u \\\\
            \\Phi_\\text{spherical} &=
                8\\pi n_m^4\\, s\\, u^2 \\\\
            \\Phi_\\text{coma} &=
                6\\pi n_m^3\\, c\\, \\eta\\, u^{3/2} \\sin(\\varphi-\\psi) \\\\
            \\Phi_\\text{astigmatism} &=
                4\\pi n_m^2\\, a\\, \\eta^2\\, u \\sin^2(\\varphi-\\psi) \\\\
            \\Phi_\\text{distortion} &=
                2\\pi n_m\\, t\\, \\eta^3\\, u^{1/2} \\sin(\\varphi-\\psi)

        where :math:`d, s, c, a, t` are ``defocus``, ``spherical``,
        ``coma``, ``astigmatism``, and ``distortion``,
        :math:`u = r^2/(r^2+z_p^2)` with :math:`r` the in-plane
        distance from the particle to the field point,
        :math:`\\eta = r_c/z_p` with :math:`r_c` the in-plane distance
        from the particle to the aberration center
        :math:`(x_c, y_c)`, and :math:`\\varphi-\\psi` is the angle
        between the field point and the aberration center, as seen
        from the particle. Defocus and spherical aberration are
        rotationally symmetric about the optical axis and do not
        depend on :math:`(x_c, y_c)`.

        Table I of [1]_ gives the generating functions and phase
        factors in a frame where :math:`\\hat{\\vec{y}}` is oriented
        along the aberration offset; the expressions above are the
        same formulas generalized to an arbitrary aberration center.

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
                     x_c: float = 0.,
                     y_c: float = 0.,
                     **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.defocus = defocus
            self.spherical = spherical
            self.coma = coma
            self.astigmatism = astigmatism
            self.distortion = distortion
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
                    'x_c': self.x_c,
                    'y_c': self.y_c}

        def _aberration(self, r_p: Coordinates) -> Field:
            '''Aberration phase mask for a particle at r_p.'''
            if not (self.defocus or self.spherical or
                    self.coma or self.astigmatism or self.distortion):
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
                dxc = r_p[0] - self.x_c
                dyc = r_p[1] - self.y_c
                rc = np.hypot(dxc, dyc)
                if rc > 0:
                    r = np.sqrt(rsq)
                    # sin(phi - psi), from sin(phi)=dy/r, cos(phi)=dx/r,
                    # sin(psi)=dyc/rc, cos(psi)=dxc/rc -- avoids calling
                    # arctan2/sin explicitly.
                    sin_rel = np.divide(dy*dxc - dx*dyc, r*rc,
                                        out=np.zeros_like(u), where=r > 0)
                    eta = rc / r_p[2]
                    root_u = np.sqrt(u)  # r / sqrt(r^2 + z_p^2)
                    if self.coma:
                        phase = phase + self.coma * (6.*np.pi*n_m**3) * \
                            eta * root_u**3 * sin_rel
                    if self.astigmatism:
                        phase = phase + self.astigmatism * \
                            (4.*np.pi*n_m**2) * eta**2 * u * sin_rel**2
                    if self.distortion:
                        phase = phase + self.distortion * \
                            (2.*np.pi*n_m) * eta**3 * root_u * sin_rel

            return np.exp(1j * phase)

        def scattered_field(self,
                            particle: Particle,
                            **kwargs) -> Field:
            '''Scattered field including geometric aberrations.'''
            field = super().scattered_field(particle, **kwargs)
            r_p = particle.r_p + particle.r_0
            return field * self._aberration(r_p)

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
