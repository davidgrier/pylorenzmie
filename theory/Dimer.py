from pylorenzmie.theory.Cluster import Cluster
from pylorenzmie.theory.Sphere import Sphere
from pylorenzmie.lib.types import Properties
from dataclasses import dataclass
import numpy as np


@dataclass
class Dimer(Cluster):
    '''A touching pair of identical spheres for Lorenz-Mie microscopy.

    Positions two :class:`~pylorenzmie.theory.Sphere` objects symmetrically
    about the cluster center, separated by twice the sphere radius.
    Orientation is specified as elevation and azimuthal angles.

    Inherits from :class:`~pylorenzmie.theory.Cluster`.

    Parameters
    ----------
    a_p : float, optional
        Radius of each sphere [μm]. Default: 0.75.
    n_p : float, optional
        Refractive index of each sphere. Default: 1.45.
    k_p : float, optional
        Imaginary part of the refractive index. Default: 0.
    θ : float, optional
        Elevation angle (θ) of the dimer axis above the xy-plane [rad].
        ``θ = 0`` → axis in the xy-plane; ``θ = π/2`` → axis along z.
        Default: π/4.
    φ : float, optional
        Azimuthal angle (φ) of the dimer axis in the xy-plane [rad].
        Default: 0.
    magnification : float, optional
        Instrument magnification [μm/pixel], used to convert the sphere
        radius to a pixel-space separation.  Default: 0.048.
    '''

    a_p: float = 0.75
    n_p: float = 1.45
    k_p: float = 0.
    θ: float = np.pi / 4.
    φ: float = 0.
    magnification: float = 0.048

    def __post_init__(self) -> None:
        p = dict(a_p=self.a_p, n_p=self.n_p, k_p=self.k_p)
        self.particles = [Sphere(**p), Sphere(**p)]
        self.update_positions()

    def __setattr__(self, key: str, value: object) -> None:
        super().__setattr__(key, value)
        if key in ('a_p', 'n_p', 'k_p'):
            for p in self.particles:
                setattr(p, key, value)
        if key in ('a_p', 'θ', 'φ', 'magnification'):
            self.update_positions()

    @Cluster.properties.getter
    def properties(self) -> Properties:
        return {'x_p': self.x_p, 'y_p': self.y_p, 'z_p': self.z_p,
                'a_p': self.a_p, 'n_p': self.n_p, 'k_p': self.k_p,
                'θ': self.θ, 'φ': self.φ}

    def update_positions(self) -> None:
        '''Set sphere positions from current orientation and radius.

        Places each sphere at ±(a_p / magnification) pixels from the
        cluster center along the dimer axis.
        '''
        if len(self.particles) != 2:
            return
        axis = np.array([np.cos(self.θ) * np.cos(self.φ),
                         np.cos(self.θ) * np.sin(self.φ),
                         np.sin(self.θ)])
        offset = (self.a_p / self.magnification) * axis
        self.particles[0].r_p = offset
        self.particles[1].r_p = -offset

    @classmethod
    def example(cls) -> None:  # pragma: no cover
        from pylorenzmie.theory import Instrument, LorenzMie
        import matplotlib.pyplot as plt

        shape = (301, 301)
        coordinates = LorenzMie.meshgrid(shape)
        instrument = Instrument()
        instrument.magnification = 0.048
        instrument.numerical_aperture = 1.45
        instrument.wavelength = 0.447
        instrument.n_m = 1.340
        dimer = cls(magnification=instrument.magnification)
        dimer.a_p = 0.5
        dimer.n_p = 1.42
        dimer.r_p = [150., 150., 250.]
        dimer.θ = np.pi / 4.
        dimer.φ = np.pi / 4.
        model = LorenzMie(coordinates=coordinates,
                          particle=dimer,
                          instrument=instrument)
        plt.imshow(model.hologram().reshape(shape), cmap='gray')
        plt.show()


if __name__ == '__main__':  # pragma: no cover
    Dimer.example()
