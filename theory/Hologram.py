from dataclasses import dataclass
from pylorenzmie.theory import (AberratedLorenzMie, LorenzMie)
from typing import Optional, Any
import numpy as np


def Hologram(cls: LorenzMie, name: str):
    '''Return class for calculating holograms'''

    @dataclass
    class HologramClass(cls):

        alpha: Optional[float] = 1.

        def __init__(self,
                     **kwargs: Optional[Any]) -> None:
            super().__init__(**kwargs)

        @cls.properties.getter
        def properties(self) -> dict:
            return {**super().properties,
                    'alpha': self.alpha}

        def hologram(self) -> np.ndarray:
            '''Return hologram of sphere

            Returns
            -------
            hologram : numpy.ndarray
                Computed hologram.
            '''
            field = self.alpha * self.field()
            field[0, :] += 1.
            hologram = np.sum(np.real(field * np.conj(field)), axis=0)
            return hologram

    HologramClass.__name__ = name
    return HologramClass


LMHologram = Hologram(LorenzMie, "LMHologram")
ALMHologram = Hologram(AberratedLorenzMie, "ALMHologram")


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

    a = ALMHologram(coordinates=coords,
                    particle=particle,
                    instrument=instrument)
    print(a)
    plt.imshow(a.hologram().reshape(shape), cmap='gray')
    plt.show()


if __name__ == '__main__':
    main()
