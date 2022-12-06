from pylorenzmie.theory.LorenzMie import (LorenzMie, example)
import numpy as np


def ALM_Factory(base_class):

    class AberratedLorenzMie(base_class):

        def __init__(self,
                     *args,
                     pupil: float = 1000.,
                     spherical: float = 0.,
                     **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.pupil = pupil
            self.spherical = spherical

        @LorenzMie.properties.getter
        def properties(self) -> dict:
            return {**super().properties,
                    'pupil': self.pupil,
                    'spherical': self.spherical}

        def aberration(self, r_p):
            '''Returns spherical aberration for particle at r_p'''
            dr = self.coordinates - r_p[:, None]
            rhosq = (dr[0]**2 + dr[1]**2) / self.pupil**2
            phase = 6.*rhosq * (rhosq - 1.) + 1.
            phase *= self.spherical
            return np.exp(-1j * phase)

        def scattered_field(self, particle, *args):
            field = super().scattered_field(particle, *args)
            r_p = particle.r_p + particle.r_0
            return field * self.aberration(r_p)

    return AberratedLorenzMie


AberratedLorenzMie = ALM_Factory(LorenzMie)


if __name__ == '__main__':
    from pylorenzmie.theory.cupyLorenzMie import cupyLorenzMie

    cupyALM = ALM_Factory(LorenzMie)
    example(cupyALM, spherical=1.)
