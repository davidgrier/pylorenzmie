from pylorenzmie.theory.LorenzMie import LorenzMie
from pylorenzmie.theory.Particle import Particle
from pylorenzmie.lib.lmtypes import Coordinates, Field, Properties
import numpy as np


def Aberrated(base_class: type) -> type:
    '''Return an aberrated subclass of a LorenzMie calculator.

    The returned class extends :meth:`scattered_field` to multiply
    the scattered field by a spherical-aberration phase mask.

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
        '''LorenzMie subclass with spherical aberration.

        Inherits from the base class supplied to :func:`Aberrated`.

        Parameters
        ----------
        spherical : float, optional
            Spherical-aberration coefficient [pixels]. Default: 0.

        Notes
        -----
        The aberration phase at each pixel is

        .. math::

            \\Phi(\\vec{r}) = s \\frac{r^4}{(r^2 + z_p^2)^2}

        where :math:`s` is ``spherical``, :math:`r` is the in-plane
        distance from the particle center in pixels, and :math:`z_p` is
        the particle's axial position in pixels.
        '''

        def __init__(self, *args,
                     spherical: float = 0.,
                     **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.spherical = spherical

        @LorenzMie.properties.getter
        def properties(self) -> Properties:
            return {**super().properties,
                    'spherical': self.spherical}

        def _aberration(self, r_p: Coordinates) -> Field:
            '''Spherical-aberration phase mask for particle at r_p.'''
            if self.spherical == 0.:
                return 1.
            x = self.coordinates[0] - r_p[0]
            y = self.coordinates[1] - r_p[1]
            rsq = x * x + y * y
            zpsq = r_p[2] ** 2
            phase = self.spherical * rsq * rsq / (rsq + zpsq) ** 2
            return np.exp(1j * phase)

        def scattered_field(self,
                            particle: Particle,
                            **kwargs) -> Field:
            '''Scattered field including spherical aberration.'''
            field = super().scattered_field(particle, **kwargs)
            r_p = particle.r_p + particle.r_0
            return field * self._aberration(r_p)

    return AberratedLorenzMie


AberratedLorenzMie = Aberrated(LorenzMie)
# Fix __qualname__ so pickle can locate the class at module level.
# Factory-defined classes get __qualname__ = 'Aberrated.<locals>.AberratedLorenzMie',
# which Python cannot resolve during unpickling.
AberratedLorenzMie.__qualname__ = 'AberratedLorenzMie'
AberratedLorenzMie.__name__ = 'AberratedLorenzMie'


if __name__ == '__main__':  # pragma: no cover
    AberratedLorenzMie.example(spherical=0.9)
