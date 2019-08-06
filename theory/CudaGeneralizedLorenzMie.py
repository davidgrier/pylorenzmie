#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from pylorenzmie.theory.GeneralizedLorenzMie import GeneralizedLorenzMie
import cupy as cp

cp.cuda.Device()

'''
This object uses generalized Lorenz-Mie theory to compute the
in-line hologram of a particle with specified Lorenz-Mie scattering
coefficients.  The hologram is calculated at specified
three-dimensional coordinates under the assumption that the
incident illumination is a plane wave linearly polarized along x.
q
REFERENCES:
1. Adapted from Chapter 4 in
   C. F. Bohren and D. R. Huffman,
   Absorption and Scattering of Light by Small Particles,
   (New York, Wiley, 1983).

2. W. J. Wiscombe, "Improved Mie scattering algorithms,"
   Appl. Opt. 19, 1505-1509 (1980).

3. W. J. Lentz, "Generating Bessel function in Mie scattering
   calculations using continued fractions," Appl. Opt. 15,
   668-671 (1976).

4. S. H. Lee, Y. Roichman, G. R. Yi, S. H. Kim, S. M. Yang,
   A. van Blaaderen, P. van Oostrum and D. G. Grier,
   "Characterizing and tracking single colloidal particles with
   video holographic microscopy," Opt. Express 15, 18275-18282
   (2007).

5. F. C. Cheong, B. Sun, R. Dreyfus, J. Amato-Grill, K. Xiao,
   L. Dixon and D. G. Grier,
   "Flow visualization and flow cytometry with holographic video
   microscopy," Opt. Express 17, 13071-13079 (2009).

HISTORY
This code was adapted from the IDL implementation of
generalizedlorenzmie__define.pro
which was written by David G. Grier.
This version is

Copyright (c) 2018 David G. Grier
'''

compute = cp.RawKernel(r'''
#include <cuComplex.h>

extern "C" __global__
void compute(float *coordsx, float *coordsy, float *coordsz,
             float x_p, float y_p, float z_p, float k,
             cuFloatComplex phase,
             float ar [], float ai [],
             float br [], float bi [],
             int norders, int length,
             bool bohren, bool cartesian,
             cuFloatComplex *e1, cuFloatComplex *e2, cuFloatComplex *e3) {
        
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x; idx < length;          idx += blockDim.x * gridDim.x) {

    double kx, ky, kz, krho, kr;
    double cosphi, costheta, coskr, sinphi, sintheta, sinkr;

    kx = k * (coordsx[idx] - x_p);
    ky = k * (coordsy[idx] - y_p);
    kz = k * (coordsz[idx] - z_p);

    kz *= -1;

    krho = sqrt(kx*kx + ky*ky);
    kr = sqrt(krho*krho + kz*kz);

    if (abs(krho) > 1e-6) {
        cosphi = kx / krho;
        sinphi = ky / krho;
    }
    else {
        cosphi = 1.0;
        sinphi = 0.0;
    }
    if (abs(kr) > 1e-6) {
        costheta = kz / kr;
        sintheta = krho / kr;
    }
    else {
        costheta = 1.0;
        sintheta = 0.0;
    }
    sincos(kr, &sinkr, &coskr);

    cuFloatComplex i = make_cuFloatComplex(0.0, 1.0);
    cuFloatComplex factor, xi_nm2, xi_nm1;
    cuFloatComplex mo1nr, mo1nt, mo1np, ne1nr, ne1nt, ne1np;
    cuFloatComplex esr, est, esp;

    if (kz > 0.) {
        factor = i;
    }
    else if (kz < 0.) {
        factor = make_cuFloatComplex(0., -1.);
    }
    else {
        factor = make_cuFloatComplex(0., 0.);
    }

    if (bohren == false) {
        factor = cuCmulf(factor, make_cuFloatComplex(-1., 0.));
    }

    xi_nm2 = cuCaddf(make_cuFloatComplex(coskr, 0.),
             cuCmulf(factor, make_cuFloatComplex(sinkr, 0.)));
    xi_nm1 = cuCsubf(make_cuFloatComplex(sinkr, 0.),
             cuCmulf(factor, make_cuFloatComplex(coskr, 0.)));

    cuFloatComplex pi_nm1 = make_cuFloatComplex(0.0, 0.0);
    cuFloatComplex pi_n = make_cuFloatComplex(1.0, 0.0);

    mo1nr = make_cuFloatComplex(0.0, 0.0);
    mo1nt = make_cuFloatComplex(0.0, 0.0);
    mo1np = make_cuFloatComplex(0.0, 0.0);
    ne1nr = make_cuFloatComplex(0.0, 0.0);
    ne1nt = make_cuFloatComplex(0.0, 0.0);
    ne1np = make_cuFloatComplex(0.0, 0.0);

    esr = make_cuFloatComplex(0.0, 0.0);
    est = make_cuFloatComplex(0.0, 0.0);
    esp = make_cuFloatComplex(0.0, 0.0);

    cuFloatComplex swisc, twisc, tau_n, xi_n, dn;

    cuFloatComplex cost, sint, cosp, sinp, krc;
    cost = make_cuFloatComplex(costheta, 0.);
    cosp = make_cuFloatComplex(cosphi, 0.);
    sint = make_cuFloatComplex(sintheta, 0.);
    sinp = make_cuFloatComplex(sinphi, 0.);
    krc  = make_cuFloatComplex(kr, 0.);

    cuFloatComplex one, two, n, fac, en, a, b;
    one = make_cuFloatComplex(1., 0.);
    two = make_cuFloatComplex(2., 0.);
    int mod;

    for (int j = 1; j < norders; j++) {
        n = make_cuFloatComplex(float(j), 0.);

        swisc = cuCmulf(pi_n, cost);
        twisc = cuCsubf(swisc, pi_nm1);
        tau_n = cuCsubf(pi_nm1, cuCmulf(n, twisc));

        xi_n = cuCsubf(cuCmulf(cuCsubf(cuCmulf(two, n), one), cuCdivf(xi_nm1, krc)), xi_nm2);

        dn = cuCsubf(cuCdivf(cuCmulf(n, xi_n), krc), xi_nm1);

        mo1nt = cuCmulf(pi_n, xi_n);
        mo1np = cuCmulf(tau_n, xi_n);

        ne1nr = cuCmulf(cuCmulf(cuCmulf(n, cuCaddf(n, one)), pi_n), xi_n);
        ne1nt = cuCmulf(tau_n, dn);
        ne1np = cuCmulf(pi_n, dn);

        mod = j % 4;
        if (mod == 1) {fac = i;}
        else if (mod == 2) {fac = make_cuFloatComplex(-1., 0.);}
        else if (mod == 3) {fac = make_cuFloatComplex(0., -1.);}
        else {fac = one;}

        en = cuCdivf(cuCdivf(cuCmulf(fac,
             cuCaddf(cuCmulf(two, n), one)), n), cuCaddf(n, one));

        a = make_cuFloatComplex(ar[j], ai[j]);
        b = make_cuFloatComplex(br[j], bi[j]);

        esr = cuCaddf(esr, cuCmulf(cuCmulf(cuCmulf(i, en), a), ne1nr));
        est = cuCaddf(est, cuCmulf(cuCmulf(cuCmulf(i, en), a), ne1nt));
        esp = cuCaddf(esp, cuCmulf(cuCmulf(cuCmulf(i, en), a), ne1np));
        esr = cuCsubf(esr, cuCmulf(cuCmulf(en, b), mo1nr));
        est = cuCsubf(est, cuCmulf(cuCmulf(en, b), mo1nt));
        esp = cuCsubf(esp, cuCmulf(cuCmulf(en, b), mo1np));

        pi_nm1 = pi_n;
        pi_n = cuCaddf(swisc, cuCmulf(cuCdivf(cuCaddf(n, one), n), twisc));

        xi_nm2 = xi_nm1;
        xi_nm1 = xi_n;
    }

    

    cuFloatComplex radialfactor = make_cuFloatComplex(1. / kr,
                                                        0.);
    cuFloatComplex radialfactorsq = make_cuFloatComplex(1. / (kr*kr),
                                                          0.);
    esr = cuCmulf(esr, cuCmulf(cuCmulf(cosp, sint), radialfactorsq));
    est = cuCmulf(est, cuCmulf(cosp, radialfactor));
    esp = cuCmulf(esp, cuCmulf(sinp, radialfactor));

    if (cartesian == true) {
        cuFloatComplex ecx, ecy, ecz;
        ecx = cuCmulf(esr, cuCmulf(sint, cosp));
        ecx = cuCaddf(ecx, cuCmulf(est, cuCmulf(cost, cosp)));
        ecx = cuCsubf(ecx, cuCmulf(esp, sinp));

        ecy = cuCmulf(esr, cuCmulf(sint, sinp));
        ecy = cuCaddf(ecy, cuCmulf(est, cuCmulf(cost, sinp)));
        ecy = cuCaddf(ecy, cuCmulf(esp, cosp));

        ecz = cuCsubf(cuCmulf(esr, cost), cuCmulf(est, sint));

        e1[idx] = ecx;
        e2[idx] = ecy;
        e3[idx] = ecz;
    }
    else {
        e1[idx] = esr;
        e2[idx] = est;
        e3[idx] = esp;
    }
    e1[idx] = cuCmulf(e1[idx], phase);
    e2[idx] = cuCmulf(e2[idx], phase);
    e3[idx] = cuCmulf(e3[idx], phase);
}
}
''', 'compute')


class CudaGeneralizedLorenzMie(GeneralizedLorenzMie):

    '''
    A class that computes scattered light fields with CUDA 
    acceleration

    ...

    Attributes
    ----------
    particle : Particle
        Object representing the particle scattering light
    instrument : Instrument
        Object resprenting the light-scattering instrument
    coordinates : numpy.ndarray
        [3, npts] array of x, y and z coordinates where field
        is calculated

    Methods
    -------
    field(cartesian=True, bohren=True)
        Returns the complex-valued field at each of the coordinates.
    '''

    def __init__(self, **kwargs):
        '''
        Parameters
        ----------
        coordinates : numpy.ndarray
           [3, npts] array of x, y and z coordinates where field
           is calculated
        particle : Particle
           Object representing the particle scattering light
        instrument : Instrument
           Object resprenting the light-scattering instrument
        n_m : complex, optional
           Refractive index of medium
        magnification : float, optional
           Magnification of microscope [um/pixel]
        wavelength : float, optional
           Vacuum wavelength of light [um]
        '''
        super(CudaGeneralizedLorenzMie, self).__init__(**kwargs)

    def _allocate(self, shape):
        '''Allocates ndarrays for calculation'''
        self.this = cp.empty(shape, dtype=np.complex64)
        self.device_coordinates = cp.asarray(self.coordinates
                                             .astype(np.float32))
        self.holo = cp.empty(shape[1], dtype=np.float32)
        self.threadsperblock = 32
        self.blockspergrid = (shape[1] + (self.threadsperblock - 1))\
            // self.threadsperblock

    def field(self, cartesian=True, bohren=True):
        '''Return field scattered by particles in the system'''
        if (self.coordinates is None or self.particle is None):
            return None
        k = np.float32(self.instrument.wavenumber())
        for p in np.atleast_1d(self.particle):
            ab = p.ab(self.instrument.n_m,
                      self.instrument.wavelength)
            a_r = ab[:, 0].real.astype(np.float32)
            a_i = ab[:, 0].imag.astype(np.float32)
            b_r = ab[:, 1].real.astype(np.float32)
            b_i = ab[:, 1].imag.astype(np.float32)
            a_r = cp.asarray(a_r)
            a_i = cp.asarray(a_i)
            b_r = cp.asarray(b_r)
            b_i = cp.asarray(b_i)
            coordsx, coordsy, coordsz = self.device_coordinates
            x_p, y_p, z_p = p.r_p.astype(np.float32)
            phase = np.complex64(np.exp(-1.j * k * z_p))
            compute((self.blockspergrid,), (self.threadsperblock,),
                    (coordsx, coordsy, coordsz,
                     x_p, y_p, z_p, k, phase,
                     a_r, a_i, b_r, b_i,
                     ab.shape[0], coordsx.shape[0],
                     cartesian, bohren,
                     *self.this))
            try:
                result += self.this
            except NameError:
                result = self.this
        return result


if __name__ == '__main__':
    from pylorenzmie.theory.FastSphere import FastSphere
    from pylorenzmie.theory.Instrument import Instrument
    import matplotlib.pyplot as plt
    from time import time
    # Create coordinate grid for image
    x = np.arange(0, 201)
    y = np.arange(0, 201)
    xv, yv = np.meshgrid(x, y)
    xv = xv.flatten()
    yv = yv.flatten()
    zv = np.zeros_like(xv)
    coordinates = np.stack((xv, yv, zv))
    # Place a sphere in the field of view, above the focal plane
    particle = FastSphere()
    particle.r_p = [150, 150, 200]
    particle.a_p = 0.5
    particle.n_p = 1.45
    # Form image with default instrument
    instrument = Instrument()
    instrument.magnification = 0.135
    instrument.wavelength = 0.447
    instrument.n_m = 1.335
    k = instrument.wavenumber()
    # Use Generalized Lorenz-Mie theory to compute field
    kernel = CudaGeneralizedLorenzMie(coordinates=coordinates,
                                      particle=particle,
                                      instrument=instrument)
    kernel.field()
    start = time()
    field = kernel.field()
    print("Time to calculate field: {}".format(time() - start))
    # Compute hologram from field and show it
    field[0, :] += 1.
    hologram = cp.sum(cp.real(field * cp.conj(field)), axis=0)
    hologram = hologram.get()
    plt.imshow(hologram.reshape(201, 201), cmap='gray')
    plt.show()
