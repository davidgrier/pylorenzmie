import numpy as np
from pylorenzmie.theory import LMHologram


def feature_extent(sphere, instrument, nfringes=20, maxrange=300):
    '''Radius of holographic feature in pixels'''

    x = np.arange(0, maxrange)
    h = LMHologram(coordinates=x, instrument=instrument)
    h.particle.a_p = sphere.a_p
    h.particle.n_p = sphere.n_p
    h.particle.z_p = sphere.z_p
    b = h.hologram() - 1.
    ndx = np.where(np.diff(np.sign(b)))[0] + 1
    extent = maxrange if (len(ndx) <= nfringes) else float(ndx[nfringes])
    return extent


if __name__ == '__main__':
    from pylorenzmie.theory import (Sphere, Instrument)

    s = Sphere(a_p=0.75, n_p=1.42, r_p=[0, 0, 200])
    n = Instrument(wavelength=0.447, n_m=1.34, magnification=0.048)
    print(feature_extent(s, n))
