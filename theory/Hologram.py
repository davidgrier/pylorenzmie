from dataclasses import dataclass
from pylorenzmie.theory import LorenzMie
import numpy as np


def Hologram(base_class: LorenzMie, name: str) -> dataclass:
    '''Return class for calculating holograms

    Arguments
    ---------
    base_class : LorenzMie
        Base class that calculates scattered field
    name : str
        Name of created hologram class

    Returns
    -------
    class : dataclass
        Class whose instances implement generative models
        for in-line holographic imaging.

    Example
    -------
    from AberratedLorenzMie import AberratedLorenzMie as base

    cls = Hologram(base, 'ALMHologram')
    a = cls()
    hologram = a.hologram()

    '''

    def hologram(self) -> np.ndarray:
        '''Return hologram of sphere

        Returns
        -------
        hologram : numpy.ndarray
            Computed hologram.
        '''
        psi = self.field()
        psi[0, :] += 1.
        hologram = np.sum(np.real(psi * np.conj(psi)), axis=0)
        return hologram

    return dataclass(type(name,
                          (base_class,),
                          {'hologram': hologram}))


LMHologram = Hologram(LorenzMie, 'LMHologram')


def run_test(cls=LMHologram, **kwargs):
    from pylorenzmie.utilities import coordinates
    from pylorenzmie.theory import (Sphere, Instrument)
    from pprint import pprint
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
    # Form image with specified instrument
    instrument = Instrument()
    instrument.magnification = 0.048
    instrument.wavelength = 0.447
    instrument.n_m = 1.340

    a = cls(coords, particle, instrument, **kwargs)

    pprint(a)
    plt.imshow(a.hologram().reshape(shape), cmap='gray')
    plt.show()


if __name__ == '__main__':
    run_test()
