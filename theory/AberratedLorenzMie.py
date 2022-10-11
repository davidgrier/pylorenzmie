from dataclasses import dataclass
from pylorenzmie.theory.LorenzMie import (LorenzMie, example)
import numpy as np


@dataclass
class AberratedLorenzMie(LorenzMie):

    pupil: float = 1000.
    spherical: float = 0.

    @LorenzMie.properties.getter
    def properties(self) -> dict:
        return {**super().properties,
                'pupil': self.pupil,
                'spherical': self.spherical}

    def scattered_field(self, particle, *args):
        psi = super().scattered_field(particle, *args)
        r_p = particle.r_p + particle.r_0
        dr = self.coordinates - r_p[:, None]
        rhosq = (dr[0]**2 + dr[1]**2) / self.pupil**2
        phi = 6.*rhosq * (rhosq - 1.) + 1.
        phi *= self.spherical
        psi *= np.exp(-1j * phi)
        return psi


if __name__ == '__main__':
    example(AberratedLorenzMie, spherical=1.)
