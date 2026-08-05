from pylorenzmie.theory.Cluster import Cluster
from pylorenzmie.theory.Sphere import Sphere
from pylorenzmie.lib.lmtypes import Properties
from dataclasses import dataclass


@dataclass
class Pair(Cluster):
    '''Two independently positioned spheres for Lorenz-Mie microscopy.

    Models two spheres with fully independent position, radius, and
    refractive index. Intended for fitting two particle groups where
    the two particles may differ in size, index, depth, and are not
    in contact (for touching particles, see
    :class:`~pylorenzmie.theory.Dimer`).

    Inherits from :class:`~pylorenzmie.theory.Cluster`.

    Parameters
    ----------
    x_p1, y_p1, z_p1 : float, optional
        Position of the first sphere, in pixels.
        Default: -10., 0., 100.
    a_p1 : float, optional
        Radius of the first sphere, in μm. Default: 1.
    n_p1 : float, optional
        Refractive index of the first sphere. Default: 1.5.
    k_p1 : float, optional
        Absorption coefficient of the first sphere. Default: 0.
    x_p2, y_p2, z_p2 : float, optional
        Position of the second sphere, in pixels.
        Default: 10., 0., 100.
    a_p2 : float, optional
        Radius of the second sphere, in μm. Default: 1.
    n_p2 : float, optional
        Refractive index of the second sphere. Default: 1.5.
    k_p2 : float, optional
        Absorption coefficient of the second sphere. Default: 0.

    Notes
    -----
    The cluster-level ``x_p``, ``y_p``, ``z_p`` inherited from
    :class:`Particle` are held fixed at the origin: sphere positions
    are absolute pixel coordinates, not offsets from a cluster center.

    :class:`~pylorenzmie.analysis.Optimizer`'s default ``fixed`` list
    names ``k_p``, which does not appear in :attr:`properties`. Pass
    ``fixed=[..., 'k_p1', 'k_p2']`` explicitly to hold both absorption
    coefficients constant during fitting.
    '''

    x_p1: float = -10.
    y_p1: float = 0.
    z_p1: float = 100.
    a_p1: float = 1.
    n_p1: float = 1.5
    k_p1: float = 0.
    x_p2: float = 10.
    y_p2: float = 0.
    z_p2: float = 100.
    a_p2: float = 1.
    n_p2: float = 1.5
    k_p2: float = 0.

    _keys = ('x_p1 y_p1 z_p1 a_p1 n_p1 k_p1 '
            'x_p2 y_p2 z_p2 a_p2 n_p2 k_p2').split()

    def __post_init__(self) -> None:
        self.x_p = self.y_p = self.z_p = 0.
        self.particles = [Sphere(), Sphere()]
        self._sync()

    def __setattr__(self, key: str, value: object) -> None:
        super().__setattr__(key, value)
        if key in Pair._keys:
            self._sync()

    @Cluster.properties.getter
    def properties(self) -> Properties:
        return {key: getattr(self, key) for key in Pair._keys}

    def _sync(self) -> None:
        '''Push x_p1..k_p2 onto the two constituent spheres.'''
        if len(self.particles) != 2:
            return
        self.particles[0].r_p = [self.x_p1, self.y_p1, self.z_p1]
        self.particles[0].a_p = self.a_p1
        self.particles[0].n_p = self.n_p1
        self.particles[0].k_p = self.k_p1
        self.particles[1].r_p = [self.x_p2, self.y_p2, self.z_p2]
        self.particles[1].a_p = self.a_p2
        self.particles[1].n_p = self.n_p2
        self.particles[1].k_p = self.k_p2

    @classmethod
    def example(cls) -> None:  # pragma: no cover
        from pylorenzmie.theory import Instrument, LorenzMie
        import matplotlib.pyplot as plt

        shape = (201, 201)
        coordinates = LorenzMie.meshgrid(shape)
        instrument = Instrument()
        instrument.magnification = 0.048
        instrument.numerical_aperture = 1.45
        instrument.wavelength = 0.447
        instrument.n_m = 1.340
        pair = cls(x_p1=85., y_p1=100., z_p1=250., a_p1=0.5, n_p1=1.42,
                  x_p2=115., y_p2=100., z_p2=250., a_p2=0.6, n_p2=1.40)
        model = LorenzMie(coordinates=coordinates,
                          particle=pair,
                          instrument=instrument)
        plt.imshow(model.hologram().reshape(shape), cmap='gray')
        plt.show()


if __name__ == '__main__':  # pragma: no cover
    Pair.example()
