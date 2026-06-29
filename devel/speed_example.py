'''
Compare time to compute 128 identical holograms across four methods:
numpy sequential, cupy sequential, torch sequential, torch batch
'''

import torch
import cupy as cp
from time import perf_counter

from pylorenzmie.theory import Sphere, Instrument
from pylorenzmie.theory.LorenzMie import LorenzMie
from pylorenzmie.theory.cupyLorenzMie import cupyLorenzMie
from pylorenzmie.theory.torchLorenzMieBatch import TorchLorenzMieBatch


def speed_example(**kwargs) -> None:
    '''
    Setup: Creates a 1024x1024 coordinate grid. Defines a list of 4
    particles as particles. Creates a list of 128 of 'particles' i.e
    128 identical holograms, each with 4 particles (multiple particles
    to properly test batching time speedup)
    '''
    N = 128
    shape = (201, 201)
    coords = TorchLorenzMieBatch.meshgrid(shape)

    instrument = Instrument()
    instrument.magnification = 0.048
    instrument.numerical_aperture = 1.45
    instrument.wavelength = 0.447
    instrument.n_m = 1.340

    def sphere(x, y, z, a_p, n_p):
        p = Sphere()
        p.r_p = [x, y, z]
        p.a_p = a_p
        p.n_p = n_p
        return p

    particles = [
        sphere(100, 100, 200, 0.5,  1.45),
        sphere(80,  120, 300, 1.0,  1.40),
        sphere(150,  50, 250, 0.75, 1.42),
        sphere(60,  160, 350, 1.2,  1.43),
    ]

    particle_lists = [particles for _ in range(N)]

    def gpu_sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # NumPy Sequential: plain lorenzmie model, loop over 128 holograms one at a time.
    numpy_model = LorenzMie(coordinates=coords, instrument=instrument)
    numpy_model.particle = particle_lists[0]
    numpy_model.hologram()  # warm-up
    t0 = perf_counter()
    for plist in particle_lists:
        numpy_model.particle = plist
        numpy_model.hologram()
    numpy_time = perf_counter() - t0

    # CuPy Sequential: same as above but using cupyLorenzMie.
    cupy_model = cupyLorenzMie(coordinates=coords, instrument=instrument)
    cupy_model.particle = particle_lists[0]
    cupy_model.hologram()  # warm-up
    cp.cuda.Device().synchronize()
    t0 = perf_counter()
    for plist in particle_lists:
        cupy_model.particle = plist
        cupy_model.hologram()
    cp.cuda.Device().synchronize()
    cupy_time = perf_counter() - t0

    # Torch Sequential: same as above but using TorchLorenzMieBatch.
    torch_model = TorchLorenzMieBatch(coordinates=coords, instrument=instrument, **kwargs)
    torch_model.particle = particle_lists[0]
    torch_model.hologram()  # warm-up
    gpu_sync()
    t0 = perf_counter()
    for plist in particle_lists:
        torch_model.particle = plist
        torch_model.hologram()
    gpu_sync()
    torch_seq_time = perf_counter() - t0

    # Torch Batch: processes 128 holograms in batches rather than serially.
    batch_size = 128
    batch_list = [particle_lists[i:i+batch_size] for i in range(0, len(particle_lists), batch_size)]

    for batch in batch_list:  # warm-up
        torch_model.batch_hologram(batch)

    gpu_sync()
    t0 = perf_counter()
    results = []
    for batch in batch_list:
        results.append(torch_model.batch_hologram(batch))
    gpu_sync()
    torch_batch_time = perf_counter() - t0

    print(f'\n{N} holograms ({shape[0]}x{shape[1]}), device: {torch_model.device}')
    print(f'{"Method":<25} {"Time (s)":>10}  {"vs numpy":>10}  {"vs prev":>10}')
    print('-' * 60)

    results = [
        ('numpy sequential',  numpy_time),
        ('cupy sequential',   cupy_time),
        ('torch sequential',  torch_seq_time),
        ('torch batch',       torch_batch_time),
    ]
    prev = numpy_time

    for name, t in results:
        print(f'{name:<25} {t:>10.3e}  {numpy_time/t:>9.1f}x  {prev/t:>9.1f}x')
        prev = t


if __name__ == '__main__':
    speed_example()
