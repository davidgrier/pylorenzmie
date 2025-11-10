from pylorenzmie.theory import (Cluster, Sphere)
from dataclasses import dataclass
import numpy as np


@dataclass
class Dimer(Cluster):

    '''
    Abstraction of a dimer of spheres for Lorenz-Mie microscopy

    Inherits
    --------
    pylorenzmie.theory.Cluster

    Properties
    ----------
    a_p : float
        Radius of each sphere in micrometers
    n_p : float
        Refractive index of each sphere
    k_p : float
        Absorption coefficient of each sphere
    theta : float
        Polar angle of the dimer axis in radians
    phi : float
        Azimuthal angle of the dimer axis in radians
    magnification : float
        Magnification of the imaging system
    '''

    a_p: float = 0.75
    n_p: float = 1.45
    k_p: float = 0.
    theta: float = np.pi/4.
    phi: float = 0.
    magnification: float = 0.048

    def __setattr__(self, key: str, value: Cluster.Property) -> None:
        super().__setattr__(key, value)
        if key in ['a_p', 'n_p', 'k_p']:
            for p in self:
                setattr(p, key, value)
        if key in ['a_p', 'theta', 'phi', 'magnification']:
            self.update_positions()

    def __post_init__(self) -> None:
        super().__post_init__()
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


def example() -> None:  # pragma: no cover
    from pylorenzmie.theory import (Instrument, LorenzMie)
    import matplotlib.pyplot as plt

    shape = (301, 301)
    coordinates = LorenzMie.meshgrid(shape)
    instrument = Instrument()
    instrument.magnification = 0.048
    instrument.numerical_aperture = 1.45
    instrument.wavelength = 0.447
    instrument.n_m = 1.340
    dimer = Dimer(magnification=instrument.magnification)
    dimer.a_p = 0.5
    dimer.n_p = 1.42
    dimer.r_p = [150., 150., 250.]
    dimer.theta = np.pi/4.
    dimer.phi = np.pi/4.
    a = LorenzMie(coordinates=coordinates,
                  particle=dimer,
                  instrument=instrument)
    plt.imshow(a.hologram().reshape(shape), cmap='gray')
    plt.show()


if __name__ == '__main__':  # pragma: no cover
    example()
