from typing import Optional
from pylorenzmie.theory.LorenzMie import LorenzMie
import numpy as np


class AberratedLorenzMie(LorenzMie):

    def __init__(self, *args,
                 pupil: Optional[float] = None,
                 spherical: Optional[float] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.pupil = pupil or 1000.
        self.spherical = spherical or 0.

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


def main():
    import matplotlib.pyplot as plt
    from pylorenzmie.utilities import coordinates
    from pylorenzmie.theory import (Sphere, Instrument)
    from time import perf_counter

    # Create coordinate grid for image
    shape = (201, 201)
    coords = coordinates(shape)
    # Place two spheres in the field of view, above the focal plane
    pa = Sphere()
    pa.r_p = [150, 150, 200]
    pa.a_p = 0.5
    pa.n_p = 1.45
    pb = Sphere()
    pb.r_p = [100, 10, 250]
    pb.a_p = 1.
    pb.n_p = 1.45
    particle = [pa, pb]
    # Form image with default instrument
    instrument = Instrument()
    instrument.magnification = 0.048
    instrument.wavelength = 0.447
    instrument.n_m = 1.340
    # Use generalized Lorenz-Mie theory to compute field
    kernel = AberratedLorenzMie(coords, particle, instrument)
    kernel.spherical = 1.
    kernel.field()
    start = perf_counter()
    field = kernel.field()
    print(f'Time to calculate: {perf_counter()-start} s')
    # Compute hologram from field and show it
    field[0, :] += 1.
    hologram = np.sum(np.real(field * np.conj(field)), axis=0)
    plt.imshow(hologram.reshape(shape), cmap='gray')
    plt.show()


if __name__ == '__main__':
    main()
