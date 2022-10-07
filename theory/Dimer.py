from pylorenzmie.theory import (Cluster, Sphere)
from dataclasses import dataclass
import numpy as np
from typing import Any


@dataclass
class Dimer(Cluster):

    a_p: float = 0.75
    n_p: float = 1.45
    k_p: float = 0.
    theta: float = np.pi/4.
    phi: float = 0.
    magnification: float = 0.048

    def __setattr__(self, key: str, value: Any) -> None:
        super().__setattr__(key, value)
        if key in ['a_p', 'n_p', 'k_p']:
            for p in self:
                setattr(p, key, value)
        if key in ['a_p', 'theta', 'phi', 'magnification']:
            self.update_positions()

    def __post_init__(self) -> None:
        p = {'a_p': self.a_p, 'n_p': self.n_p, 'k_p': self.k_p}
        self.particles = [Sphere(**p), Sphere(**p)]
        self.update_positions()

    def update_positions(self) -> None:
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

    shape = (301, 301)
    coords = coordinates(shape)
    instrument = Instrument()
    instrument.magnification = 0.048
    instrument.wavelength = 0.447
    instrument.n_m = 1.340
    dimer = Dimer(magnification=instrument.magnification)
    dimer.a_p = 0.5
    dimer.n_p = 1.42
    dimer.r_p = [150., 150., 250.]
    dimer.theta = 0.
    dimer.phi = np.pi/4.

    a = LMHologram(coordinates=coords,
                   particle=dimer,
                   instrument=instrument)
    plt.imshow(a.hologram().reshape(shape), cmap='gray')
    plt.show()


if __name__ == '__main__':
    main()
