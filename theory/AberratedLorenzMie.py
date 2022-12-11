from pylorenzmie.theory.LorenzMie import (LorenzMie, example)
from pylorenzmie.theory.Particle import Particle
import numpy as np
from typing import (Tuple, Dict, Any)


def ALM_Factory(base_class: LorenzMie):

    '''Returns a class definition for a Lorenz-Mie theory
    that incoporates spherical aberration

    Arguments
    ---------
    base_class : LorenzMie
        Scattering theory to be subclassed
    '''

    class AberratedLorenzMie(base_class):

        '''Class definition for a Lorenz-Mie theory that
        incorporates spherical aberration.

        Inherits
        --------
        LorenzMie

        Properties
        ----------
        spherical : float
            Amount of spherical aberration to incorporate.
        '''

        def __init__(self,
                     *args: Tuple[Any],
                     spherical: float = 0.,
                     **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.spherical = spherical

        @LorenzMie.properties.getter
        def properties(self) -> Dict:
            return {**super().properties,
                    'spherical': self.spherical}

        def _aberration(self, r_p: np.ndarray) -> Any:
            '''Returns spherical aberration for particle at r_p'''
            NA = self.instrument.numerical_aperture
            n_m = self.instrument.n_m
            omega = 2.*NA*r_p[2]/n_m
            x = (self._device_coordinates[0] - r_p[0]) / omega
            y = (self._device_coordinates[1] - r_p[1]) / omega
            rhosq = x*x + y*y
            phase = 6.*rhosq * (rhosq - 1.) + 1.
            phase *= -self.spherical
            return self.to_field(phase)

        def scattered_field(self,
                            particle: Particle,
                            *args: Tuple[Any]):
            field = super().scattered_field(particle, *args)
            r_p = particle.r_p + particle.r_0
            return field * self._aberration(r_p)

    return AberratedLorenzMie


AberratedLorenzMie = ALM_Factory(LorenzMie)


if __name__ == '__main__':
    example(AberratedLorenzMie, spherical=0.9)
