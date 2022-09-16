from pylorenzmie.theory import Cluster, Sphere
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Dimer(Cluster):

    particles: list = field(default_factory=lambda: [Sphere(), Sphere()])
    a_p: float = 1.
    n_p: float = 1.5
    k_p: float = 0.
    theta: float = 0.
    phi: float = 0.
    magnification: float = 0.048

    def __setattr__(self, key, value):
        super().__setattr__(key, value)
        self.update_particles(key, value)
        if key in ['a_p', 'theta', 'phi']:
            self.update_positions()

    def _initialized(self):
        return hasattr(self, 'particles') and (len(self) == 2)

    def update_particles(self, key, value):
        if hasattr(self, 'particles'):
            for p in self.particles:
                if hasattr(p, key):
                    setattr(p, key, value)

    def update_positions(self):
        if len(self) != 2:
            return
        x_p = np.cos(self.theta) * np.cos(self.phi)
        y_p = np.cos(self.theta) * np.sin(self.phi)
        z_p = np.sin(self.theta)
        r_p = (self.a_p/self.magnification) * np.array([x_p, y_p, z_p])
        self.particles[0].r_p = r_p
        self.particles[1].r_p = -r_p


def main():
    from pylorenzmie.utilities import coordinates
    from pylorenzmie.theory import Instrument, LMHologram
    import matplotlib.pyplot as plt

    shape = (201, 201)
    coords = coordinates(shape)
    instrument = Instrument()
    instrument.magnification = 0.048
    instrument.wavelength = 0.447
    instrument.n_m = 1.340
    dimer = Dimer(magnification=instrument.magnification)
    dimer.r_p = [100., 100., 100.]
    for p in dimer:
        print(p)

    a = LMHologram(coordinates=coords,
                   particle=dimer,
                   instrument=instrument)
    plt.imshow(a.hologram().reshape(shape), cmap='gray')
    plt.show()


if __name__ == '__main__':
    main()
