from pylorenzmie.theory import LorenzMie
import numpy as np


class AberratedLorenzMie(LorenzMie):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
            dr = self.coordinates - p.r_p[:, None]
            self.krv[...] = np.asarray(k * dr)
            ab = p.ab(n_m, wavelength)
            this = self.compute(ab, self.krv, *self.buffers,
                                cartesian=cartesian, bohren=bohren)
            ### COMPUTE ABERRATIONS FOR PARTICLE
            ### this *= aberrations
            this *= np.exp(-1j * k * p.z_p)
            self.result += this
        return self.result
