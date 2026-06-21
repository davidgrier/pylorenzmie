from pylorenzmie.theory.LorenzMie import LorenzMie
from pylorenzmie.theory.Particle import Particle
from pylorenzmie.lib.types import Coordinates, Field, Properties
import numpy as np


def Aberrated(base_class: type) -> type:
    '''Return an aberrated subclass of a LorenzMie calculator.

    The returned class extends :meth:`scattered_field` to multiply
    the scattered field by a Zernike spherical-aberration phase mask
    evaluated at the pupil plane.

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
        '''LorenzMie subclass with Zernike spherical aberration.

        Inherits from the base class supplied to :func:`Aberrated`.

        Parameters
        ----------
        spherical : float, optional
            Zernike spherical-aberration coefficient [waves].
            Default: 0 (no aberration).
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

        def _aperture(self, z_p: float) -> float:
            NA = self.instrument.numerical_aperture
            n_m = self.instrument.n_m
            return 2. * NA * z_p / n_m

        def _aberration(self, r_p: Coordinates) -> Field:
            '''Zernike spherical-aberration phase mask for particle at r_p.'''
            omega = self._aperture(r_p[2])
            if omega == 0. or self.spherical == 0.:
                return 1.
            x = (self.coordinates[0] - r_p[0]) / omega
            y = (self.coordinates[1] - r_p[1]) / omega
            rhosq = x * x + y * y
            phase = 6. * rhosq * (rhosq - 1.) + 1.
            phase *= -self.spherical
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


if __name__ == '__main__':  # pragma: no cover
    AberratedLorenzMie.example(spherical=0.9)
