from collections.abc import Iterator
from dataclasses import dataclass, field
import numpy as np
from pylorenzmie.lib import LMObject
from pylorenzmie.lib.types import (Coordinates, Coefficients, Properties)


@dataclass
class Particle(LMObject):
    '''Abstract base for a scattering particle in Lorenz-Mie microscopy.

    Holds the particle's 3-D position and the origin of the local
    coordinate system.  Subclasses add the physical parameters needed
    to compute Mie scattering coefficients via :meth:`ab`.

    ``Particle`` implements the sequence protocol (``__len__``,
    ``__iter__``, ``__getitem__``) so that a single particle and a list
    of particles are interchangeable in :meth:`LorenzMie.field`.

    Attributes
    ----------
    x_p : float
        x coordinate of the particle center, in pixels. Default: 0.
    y_p : float
        y coordinate of the particle center, in pixels. Default: 0.
    z_p : float
        z coordinate of the particle center, in pixels. Default: 100.
    x_0 : float
        x coordinate of the local origin, in pixels. Default: 0.
    y_0 : float
        y coordinate of the local origin, in pixels. Default: 0.
    z_0 : float
        z coordinate of the local origin, in pixels. Default: 0.

    Notes
    -----
    All coordinates are in pixels.  ``r_0`` is set by :class:`Cluster`
    to translate each constituent particle into the cluster's local
    frame; it is not an adjustable parameter and is excluded from
    ``properties``.
    '''

    x_p: float = 0.
    y_p: float = 0.
    z_p: float = 100.
    x_0: float = field(repr=False, default=0.)
    y_0: float = field(repr=False, default=0.)
    z_0: float = field(repr=False, default=0.)

    def __len__(self) -> int:
        return 1

    def __iter__(self) -> Iterator['Particle']:
        return iter([self])

    def __getitem__(self, index: int) -> 'Particle':
        if index != 0:
            raise IndexError('Particle index out of range')
        return self

    @property
    def r_p(self) -> Coordinates:
        '''3-D position of the particle center, in pixels.

        Returns
        -------
        numpy.ndarray, shape (3,)
            ``[x_p, y_p, z_p]``.
        '''
        return np.asarray([self.x_p, self.y_p, self.z_p])

    @r_p.setter
    def r_p(self, r_p: Coordinates) -> None:
        self.x_p, self.y_p, self.z_p = r_p

    @property
    def r_0(self) -> Coordinates:
        '''Origin of the local coordinate system, in pixels.

        Returns
        -------
        numpy.ndarray, shape (3,)
            ``[x_0, y_0, z_0]``.
        '''
        return np.asarray([self.x_0, self.y_0, self.z_0])

    @r_0.setter
    def r_0(self, r_0: Coordinates) -> None:
        self.x_0, self.y_0, self.z_0 = r_0

    @LMObject.properties.getter
    def properties(self) -> Properties:
        return {'x_p': self.x_p,
                'y_p': self.y_p,
                'z_p': self.z_p}

    def ab(self,
           n_m: float | complex = 1.+0.j,
           wavelength: float = 0.532) -> Coefficients:
        '''Mie scattering coefficients for this particle.

        The base-class implementation returns ``ones((1, 2))`` as a
        trivial placeholder.  Subclasses must override this method with
        a physically meaningful calculation.

        Parameters
        ----------
        n_m : float or complex
            Refractive index of the surrounding medium. Default: 1+0j.
        wavelength : float
            Vacuum wavelength of the illuminating light, in μm.
            Default: 0.532.

        Returns
        -------
        ab : numpy.ndarray, shape (n_terms, 2), dtype complex
            Mie scattering coefficients.
        '''
        return np.ones((1, 2), dtype=complex)


if __name__ == '__main__':  # pragma: no cover
    Particle.example()
