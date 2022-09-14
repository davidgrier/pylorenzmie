from typing import Optional
from pylorenzmie.theory import LorenzMie
import numpy as np


class AberratedLorenzMie(LorenzMie):

    def __init__(self, *args,
                 pupil: Optional[float] = None,
                 spherical: Optional[float] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.pupil = pupil or 1000.
        self.spherical = spherical or 0.

    def field(self,
              cartesian: bool = True,
              bohren: bool = True) -> np.ndarray:
        '''Return field scattered by particles in the system'''
        if (self.coordinates is None or self.particle is None):
            return None
        k = self.instrument.wavenumber()
        n_m = self.instrument.n_m
        wavelength = self.instrument.wavelength
        self.result.fill(0.+0.j)
        for p in np.atleast_1d(self.particle):
            dr = self.coordinates - p.r_p[:, None] - p.r_0[:, None]
            # scattered field
            self.krv[...] = np.asarray(k * dr)
            ab = p.ab(n_m, wavelength)
            this = self.compute(ab, self.krv, *self.buffers,
                                cartesian=cartesian, bohren=bohren)
            # aberrations

            psi = self.aberration(dr)
            # NOTE: this.shape = (3, npts)
            #       psi.shape = (npts)
            #       How should I multiply this * psi so that
            #       each dimension of this is multiplied by psi?
            this[0] *= psi
            this[1] *= psi
            this[2] *= psi
            
            # overall phase
            this *= np.exp(-1j * k * p.z_p)
            self.result += this
        return self.result

    def aberration(self, dr: np.ndarray) -> np.ndarray:
        rhosq = (dr[0]**2 + dr[1]**2) / self.pupil**2
        phi = 6.*rhosq * (rhosq - 1.) + 1.
        phi *= self.spherical  # any dependence on z_p (dr[2]) goes here.
        return np.exp(-1j * phi)


def main():
    import matplotlib.pyplot as plt
    from pylorenzmie.utilities import coordinates
    from pylorenzmie.theory import (Sphere, Instrument)
    from time import perf_counter

    # Create coordinate grid for image
    coords = coordinates((201, 201))
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
    kernel.spherical = 0.*np.pi
    kernel.field()
    start = perf_counter()
    field = kernel.field()
    print(f'Time to calculate: {perf_counter()-start} s')
    # Compute hologram from field and show it
    field[0, :] += 1.
    hologram = np.sum(np.real(field * np.conj(field)), axis=0)
    plt.imshow(hologram.reshape(201, 201), cmap='gray')
    plt.show()


if __name__ == '__main__':
    main()
