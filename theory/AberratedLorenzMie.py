from pylorenzmie.theory.LorenzMie import (LorenzMie, example)
import numpy as np


def ALM_Factory(base_class):

    class AberratedLorenzMie(base_class):

        def __init__(self,
                     *args,
                     spherical: float = 0.,
                     **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.spherical = spherical

        @LorenzMie.properties.getter
        def properties(self) -> dict:
            return {**super().properties,
                    'spherical': self.spherical}

        @property
        def pupil(self):
            NA = self.instrument.numerical_aperture
            n_m = self.instrument.n_m
            return 2.*NA/n_m

        def aberration(self, r_p):
            '''Returns spherical aberration for particle at r_p'''
            omega = self.pupil * r_p[2]
            x = (self._device_coordinates[0] - r_p[0]) / omega
            y = (self._device_coordinates[1] - r_p[1]) / omega
            rhosq = x*x + y*y
            phase = 6.*rhosq * (rhosq - 1.) + 1.
            phase *= -self.spherical
            return self.to_field(phase)

        def scattered_field(self, particle, *args):
            field = super().scattered_field(particle, *args)
            r_p = particle.r_p + particle.r_0
            return field * self.aberration(r_p)

    return AberratedLorenzMie


AberratedLorenzMie = ALM_Factory(LorenzMie)


if __name__ == '__main__':
    example(AberratedLorenzMie, spherical=0.9)
