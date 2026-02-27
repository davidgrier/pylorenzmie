'''
Example using torchLorenzmie.py, following the same setup as the example in LorenzMie.py
'''

import torch
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
from types import SimpleNamespace

from pylorenzmie.theory import Sphere, Instrument
from torchLorenzMie import LorenzMieBatch, field
from pylorenzmie.lib import LMObject
from pylorenzmie.theory.LorenzMie import LorenzMie

def example():
    instrument = Instrument()
    instrument.magnification = 0.048
    instrument.numerical_aperture = 1.45
    instrument.wavelength = 0.447
    instrument.n_m = 1.340

    pa = Sphere()
    pa.r_p = [150, 150, 200]
    pa.a_p = 0.5
    pa.n_p = 1.45

    pb = Sphere()
    pb.r_p = [100, 10, 250]
    pb.a_p = 1.
    pb.n_p = 1.45

    n_m        = instrument.n_m
    wavelength = instrument.wavelength

    particles = []
    for sphere in [pa,pb]:
        ab  = sphere.ab(n_m, wavelength)
        r_p = sphere.r_p + sphere.r_0
        p   = SimpleNamespace(
            r_p = torch.tensor(r_p, dtype=torch.float32),
            ab  = torch.tensor(ab,  dtype=torch.complex64),
        )
        particles.append(p)

    shape = (1024, 1280)
#    print(f'{ab.shape = }')

    gen   = LorenzMieBatch(
        shape         = shape,
        wavelength    = instrument.wavelength,
        n_m           = instrument.n_m,
        magnification = instrument.magnification,
        NA            = instrument.numerical_aperture,
        spherical     = 0.,   # no aberration for direct comparison with numpy
        device        = None
    )
    
    start = perf_counter()
    holo  = gen.hologram(particles).reshape(shape)
    print(f'Time to calculate: {perf_counter()-start:.1e} s')

    start = perf_counter()
    holo  = gen.hologram(particles).reshape(shape)
    print(f'Second pass: {perf_counter()-start:.1e} s')

    holo_np = holo.cpu().numpy()
    plt.figure(num='LorenzMieBatch example')
    plt.imshow(holo_np, cmap='gray')
#    plt.colorbar()
    plt.title('Torch hologram')
#    plt.show()
    plt.savefig('torchhologram.png')

if __name__ == '__main__':
    example()

