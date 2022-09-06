from pylorenzmie.lib import LMObject
from pylorenzmie.theory import LorenzMie
from typing import Optional, Any
import numpy as np


class LMHologram(LMObject):
    '''
    Compute in-line hologram of a sphere

    ...

    Properties
    ----------
    alpha : float
        Relative amplitude of scattered field.
        Default: 1

    kernel : LorenzMie
        Kernel for calculating scattered fields

    Methods
    -------
    hologram() : numpy.ndarray
        Computed hologram of sphere
    '''

    def __init__(self,
                 alpha: Optional[float] = None,
                 kernel: Optional[LorenzMie] = None,
                 **kwargs: Optional[Any]) -> None:
        self.alpha = alpha or 1.
        self.kernel = kernel or LorenzMie(**kwargs)

    def __str__(self) -> str:
        fmt = '<{}(alpha={})>'
        return fmt.format(self.__class__.__name__, self.alpha)

    def __repr__(self) -> str:
        return str(self)

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        self._alpha = value

    @property
    def properties(self) -> dict:
        p = self.kernel.properties
        p['alpha'] = self.alpha
        return p

    @properties.setter
    def properties(self, p: dict) -> None:
        self.kernel.properties = p
        for key, value in p.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def hologram(self) -> np.ndarray:
        '''Return hologram of sphere

        Returns
        -------
        hologram : numpy.ndarray
            Computed hologram.
        '''
        field = self.alpha * self.kernel.field()
        field[0, :] += 1.
        hologram = np.sum(np.real(field * np.conj(field)), axis=0)
        return hologram


def main():
    from pylorenzmie.utilities import coordinates
    from pylorenzmie.theory import (Sphere, Instrument)
    import matplotlib.pyplot as plt

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

    a = LMHologram(coordinates=coords,
                   particle=particle,
                   instrument=instrument)
    print(a)
    plt.imshow(a.hologram().reshape(shape), cmap='gray')
    plt.show()


if __name__ == '__main__':
    main()
