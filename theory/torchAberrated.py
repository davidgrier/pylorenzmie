'''
Torch implemenation of aberration functions for LM holograms

This code computes the spherical aberration phase correction for each particle that is passed through it and returns the corrected field for that particle
'''
import torch

# converted the AberratedLorenzMie._aberration() to tensor format
def spherical_aberration(coordinates, r_p, NA, n_m, spherical):
    omega = 2. * NA * r_p[2] / n_m
    x = (coordinates[0] - r_p[0]) / omega
    y = (coordinates[1] - r_p[1]) / omega
    rhosq = x*x + y*y
    phase = 6. * rhosq * (rhosq - 1.) + 1.
    phase *= -spherical
    return torch.exp(1j * phase.to(torch.complex64))
