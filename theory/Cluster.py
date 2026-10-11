'''A group of particles that is analyzed as one scatterer.

A :class:`Cluster` holds a list of particles, such as spheres, and has a
position of its own that is the **origin of the coordinates of its members**:
the position of each member is relative to the cluster, so a sphere at
``r_p = (10, 0, 50)`` in a cluster at ``(100, 100, 100)`` is at
``(110, 100, 150)`` in the image.  By default the cluster is at
``z_p = 100`` pixels, not at the focal plane; set ``x_p``, ``y_p`` and ``z_p``
to zero to use absolute coordinates.

The calculators sum the fields of the members coherently.  Only the position
of the cluster is a fit parameter (``properties``); the members keep their
own.  :class:`~pylorenzmie.theory.Dimer.Dimer` is a cluster of two touching
spheres.
'''

from pylorenzmie.theory.Particle import Particle
from pylorenzmie.lib.lmtypes import Properties
from dataclasses import dataclass, field


@dataclass
class Cluster(Particle):
    '''A cluster of particles for Lorenz-Mie microscopy.

    Groups a list of :class:`~pylorenzmie.theory.Particle.Particle` objects
    into a single scatterer.  The cluster position ``r_p`` sets the origin of
    each constituent particle's coordinate system via
    :attr:`~pylorenzmie.theory.Particle.Particle.r_0`.

    Inherits from :class:`~pylorenzmie.theory.Particle`.

    Parameters
    ----------
    particles : list of Particle, optional
        Constituent particles.  Default: empty list.

    Notes
    -----
    Setting ``x_p``, ``y_p``, ``z_p``, or ``particles`` automatically
    calls :meth:`update` to propagate the new cluster center to each
    constituent particle's ``r_0``.
    '''

    particles: list[Particle] = field(repr=False, default_factory=list)

    def __post_init__(self) -> None:
        self.update()

    def __setattr__(self, key: str, value: object) -> None:
        super().__setattr__(key, value)
        if key in ('x_p', 'y_p', 'z_p', 'particles'):
            self.update()

    def __len__(self) -> int:
        return len(self.particles)

    def __iter__(self):
        return iter(self.particles)

    def __getitem__(self, index: int) -> Particle:
        return self.particles[index]

    @Particle.properties.getter
    def properties(self) -> Properties:
        return {'x_p': self.x_p,
                'y_p': self.y_p,
                'z_p': self.z_p}

    def update(self) -> None:
        '''Propagate cluster position to each constituent particle's origin.'''
        try:
            for particle in self.particles:
                particle.r_0 = self.r_p
        except AttributeError:
            pass
