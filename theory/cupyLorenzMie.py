#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pylorenzmie.theory.LorenzMie import (LorenzMie, example)
from pylorenzmie.theory import Particle
from typing import Union
import numpy as np
import cupy as cp


class cupyLorenzMie(LorenzMie):
    '''
    Compute scattered light field with CUDA acceleration.

    ...

    Properties
    ----------
    double_precision : bool
        If True, perform GPU calculations in double precision.
        Default: True

    Methods
    -------
    field(cartesian=True, bohren=True, gpu=False)
        Returns the complex-valued field at each of the coordinates.

        gpu : bool
            If True, return gpu variable for field.
            Default [False]: return cpu field

    '''

    method: str = 'cupy'

    def __init__(self,
                 *args,
                 double_precision: bool = True,
                 **kwargs):
        self.ctype = None
        super().__init__(*args, **kwargs)
        self.double_precision = double_precision

    @property
    def double_precision(self) -> bool:
        return self._double_precision

    @double_precision.setter
    def double_precision(self, double_precision: bool) -> None:
        # NOTE: Check if GPU is capable of double precision
        self._double_precision = double_precision
        if double_precision:
            self.kernel = self.cufield()
            self.dtype = np.float64
            self.ctype = np.complex128
        else:
            self.kernel = self.cufieldf()
            self.dtype = np.float32
            self.ctype = np.complex64
        self.allocate()

    def to_field(self, phase):
        return cp.exp(1j * phase)

    def scattered_field(self,
                        particle: Particle,
                        cartesian: bool,
                        bohren: bool) -> None:
        ab = particle.ab(self.instrument.n_m, self.instrument.wavelength)
        ar = ab[:, 0].real.astype(self.dtype)
        ai = ab[:, 0].imag.astype(self.dtype)
        br = ab[:, 1].real.astype(self.dtype)
        bi = ab[:, 1].imag.astype(self.dtype)
        ar, ai, br, bi = cp.asarray([ar, ai, br, bi])
        r_p = (particle.r_p + particle.r_0).astype(self.dtype)
        k = self.dtype(self.instrument.wavenumber())
        phase = np.exp(-1.j * k * r_p[2], dtype=self.ctype)
        self.kernel((self.blockspergrid,), (self.threadsperblock,),
                    (*self.gpu_coordinates, *r_p, k, phase,
                     ar, ai, br, bi, ab.shape[0],
                     self.gpu_coordinates.shape[1],
                     cartesian, bohren,
                     *self.buffer))
        return self.buffer

    def field(self, **kwargs) -> np.ndarray:
        return self._device_field(**kwargs).get()

    @property
    def _device_coordinates(self):
        return self.gpu_coordinates

    def _device_field(self,
                      cartesian: bool = True,
                      bohren: bool = True) -> cp.ndarray:
        '''Return field scattered by particles in the system'''
        if (self.coordinates is None or self.particle is None):
            return None
        self.result.fill(0.+0.j)
        for p in np.atleast_1d(self.particle):
            self.result += self.scattered_field(p, cartesian, bohren)
        return self.result

    def allocate(self) -> None:
        '''Allocate buffers for calculation'''
        if (self.coordinates is None) or (self.ctype is None):
            return
        shape = self.coordinates.shape
        self.result = cp.empty(shape, dtype=self.ctype)
        self.buffer = cp.empty(shape, dtype=self.ctype)
        self.gpu_coordinates = cp.asarray(self.coordinates, self.dtype)
        self.holo = cp.empty(shape[1], dtype=self.dtype)
        self.threadsperblock = 32
        self.blockspergrid = ((shape[1] + (self.threadsperblock - 1)) //
                              self.threadsperblock)

    def cufieldf(self) -> cp.RawKernel:
        '''Return CUDA kernel for single-precision field computation'''
        return cp.RawKernel(self.kernel, 'field')

    def cufield(self) -> cp.RawKernel:
        '''Return CUDA kernel for double-precision field computation'''
        change = {'f(': '(', 'float': 'double', 'Float': 'Double'}
        kernel = self.kernel
        for before, after in change.items():
            kernel = kernel.replace(before, after)
        return cp.RawKernel(kernel, 'field')

    kernel = r'''
#include <cuComplex.h>

extern "C" __global__
void field(float *coordsx, float *coordsy, float *coordsz,
           float x_p, float y_p, float z_p, float k,
           cuFloatComplex phase,
           float ar [], float ai [],
           float br [], float bi [],
           int norders, int length,
           bool bohren, bool cartesian,
           cuFloatComplex *e1, cuFloatComplex *e2, cuFloatComplex *e3) {

    for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
         idx < length;
         idx += blockDim.x * gridDim.x) {

        float kx, ky, kz, krho, kr, phi, theta;
        float cosphi, costheta, coskr, sinphi, sintheta, sinkr;

        kx = k * (coordsx[idx] - x_p);
        ky = k * (coordsy[idx] - y_p);
        kz = -k * (coordsz[idx] - z_p);

        krho = sqrt(kx*kx + ky*ky);
        kr = sqrt(krho*krho + kz*kz);

        phi = atan2(ky, kx);
        theta = atan2(krho, kz);
        sincos(phi, &sinphi, &cosphi);
        sincos(theta, &sintheta, &costheta);
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
            factor = cuCmulf(factor, make_cuFloatComplex(-1.,  0.));
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

            xi_n = cuCsubf(cuCmulf(cuCsubf(cuCmulf(two, n), one),
                           cuCdivf(xi_nm1, krc)), xi_nm2);

            dn = cuCsubf(cuCdivf(cuCmulf(n, xi_n), krc), xi_nm1);

            mo1nt = cuCmulf(pi_n, xi_n);
            mo1np = cuCmulf(tau_n, xi_n);

            ne1nr = cuCmulf(cuCmulf(cuCmulf(n, cuCaddf(n, one)),
                            pi_n), xi_n);
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
            pi_n = cuCaddf(swisc,
                           cuCmulf(cuCdivf(cuCaddf(n, one), n), twisc));

            xi_nm2 = xi_nm1;
            xi_nm1 = xi_n;
        }

        cuFloatComplex radialfactor = make_cuFloatComplex(1./kr, 0.);
        cuFloatComplex radialfactorsq = make_cuFloatComplex(1./(kr*kr), 0.);
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

            e1[idx] = cuCmulf(ecx, phase);
            e2[idx] = cuCmulf(ecy, phase);
            e3[idx] = cuCmulf(ecz, phase);
        }
        else {
            e1[idx] = cuCmulf(esr, phase);
            e2[idx] = cuCmulf(est, phase);
            e3[idx] = cuCmulf(esp, phase);
        }
    }
}
'''


if __name__ == '__main__':
    example(cupyLorenzMie, double_precision=True)
